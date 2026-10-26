using System;
using System.Collections.Generic;
using UnityEngine;

namespace SPWM.Robotics
{
    public enum ActionControlMode
    {
        MotorTargetPD = 0,
        DirectJointTorque = 1
    }

    /// <summary>
    /// Telemetry and execution phase flags for SPWM-v0.5.0 6-DOF Active Perception Grasping:
    ///   0 = BlindApproach  (includes Phase 0 neutral resting start + Phase 1 deliberate approach to ground object)
    ///   1 = SensoryHeft    (Phase 1.5 instant rigid lock + sharp deliberate upward lift to feel the object's weight)
    ///   2 = BlindTransport (Phase 2 deliberate transport to a new 3D location while twisting wrist joints upright)
    /// </summary>
    public enum GraspTrajectoryPhase
    {
        BlindApproach = 0,
        SensoryHeft = 1,
        BlindTransport = 2
    }

    /// <summary>
    /// Sequential Waypoint PD Controller for the 6-DOF Industrial Manipulator (SPWM-v0.5.0).
    /// Outputs 6-dimensional continuous motor vectors u_t in [-1, 1]^6 (normalized by MaxJointAngleRad = 2.60 rad)
    /// executing a deliberate 4-stage industrial grasping workflow without any oscillatory or walking artifacts:
    ///   - Phase 0 (The Neutral Start): Begins every episode at a clean, poised neutral resting posture.
    ///   - Phase 1 (The Approach, flag 0): Sweeps the end-effector tip smoothly and deliberately to the exact
    ///     3D grasp point on the top surface of the target object sitting on the ground.
    ///   - Phase 1.5 (The "Heft", flag 1): Locks onto the object via an instant rigid FixedJoint (like a magnet)
    ///     and immediately performs a sharp, deliberate upward lift (+12 cm) with a crisp wrist twist so the
    ///     added weight and rotational inertia of the heavy object physically pull on the arm's joints.
    ///   - Phase 2 (The Transport, flag 2): Holding the object tightly, carries it smoothly to a new 3D workspace
    ///     location while deliberately twisting the wrist joints (Pitch, Yaw, Roll) to reorient the object
    ///     from flat on the ground to upright.
    /// </summary>
    [RequireComponent(typeof(ArticulatedRobotRig))]
    public class ActionConditionedController : MonoBehaviour
    {
        [Header("PD Control & Physics Parameters")]
        [SerializeField] private ActionControlMode controlMode = ActionControlMode.MotorTargetPD;
        [SerializeField] private float proportionalGainKp = 0.85f;
        [SerializeField] private float derivativeGainKd = 0.18f;
        [SerializeField] private float viscousJointDamping = 0.15f;
        [SerializeField] private float maxMotorTorque = 240.0f;
        [SerializeField] private float targetAngleScaleRad = ArticulatedRobotRig.MaxJointAngleRad;
        [SerializeField] private Vector3 gravityVector = new Vector3(0f, -9.81f, 0f);

        [Header("Phasic Waypoint Timing (Steps)")]
        [SerializeField] private int setupHoldSteps = 6;
        [SerializeField] private int blindApproachSteps = 34;
        [SerializeField] private int sensoryHeftSteps = 12;

        [Header("Sensory Heft Sharp Upward Lift")]
        [SerializeField] private float heftLiftMeters = 0.12f;
        [SerializeField] private Vector3 heftWristTwistRad = new Vector3(0.24f, -0.28f, 0.36f);
        [SerializeField] private float heftAccelerationBoost = 2.2f;

        private ArticulatedRobotRig _rig;
        private ProceduralObjectSpawner _spawner;
        private float[] _currentActionVector;

        private float[] _qNeutral;
        private float[] _qGrasp;
        private float[] _qHeftPeak;
        private float[] _qTarget6D;
        private SpawnedPayloadSpec _activeEpisodeSpec;
        private GraspTrajectoryPhase _currentPhase = GraspTrajectoryPhase.BlindApproach;

        public int ActionDim => _rig != null ? _rig.JointCount : ArticulatedRobotRig.CanonicalJointCount;
        public float[] CurrentAction => _currentActionVector;
        public GraspTrajectoryPhase CurrentPhase => _currentPhase;
        public int CurrentPhaseFlag => (int)_currentPhase;
        public SpawnedPayloadSpec ActiveEpisodeSpec => _activeEpisodeSpec;

        private void Awake()
        {
            _rig = GetComponent<ArticulatedRobotRig>();
            _spawner = GetComponent<ProceduralObjectSpawner>();
            Physics.gravity = gravityVector;
            targetAngleScaleRad = ArticulatedRobotRig.MaxJointAngleRad;
            InitializeControllerState(seed: 42);
        }

        /// <summary>
        /// Computes the exact 6-DOF joint-space waypoints:
        ///   _qNeutral  -> Clean neutral resting posture
        ///   _qGrasp    -> Exact 3D approach to the top surface of the object sitting on the ground
        ///   _qHeftPeak -> Sharp +12 cm upward lift + crisp wrist twist immediately after rigid magnetic lock
        ///   _qTarget6D -> New 3D workspace location with wrist joints twisted to orient the object upright
        /// </summary>
        public void InitializeControllerState(int seed, SpawnedPayloadSpec? payloadSpec = null)
        {
            int dof = _rig != null && _rig.JointCount > 0 ? _rig.JointCount : ArticulatedRobotRig.CanonicalJointCount;
            _currentActionVector = new float[dof];
            _currentPhase = GraspTrajectoryPhase.BlindApproach;
            targetAngleScaleRad = ArticulatedRobotRig.MaxJointAngleRad;

            _qNeutral = ArticulatedRobotRig.GetNeutralRestingPose();
            Vector3 baseOrigin = _rig != null ? _rig.transform.position : transform.position;

            SpawnedPayloadSpec spec;
            if (payloadSpec.HasValue)
            {
                spec = payloadSpec.Value;
            }
            else if (_spawner != null)
            {
                spec = _spawner.SpawnEpisodePayload(seed, _rig);
            }
            else
            {
                spec = ProceduralObjectSpawner.GenerateDeterministicSpec(seed, baseOrigin);
            }
            _activeEpisodeSpec = spec;

            // 1. Approach Waypoint: End-effector tip reaches the exact top surface of the ground object
            Vector3 graspPoint = spec.GraspPoint;
            _qGrasp = _rig != null
                ? _rig.SolveInverseKinematics6D(graspPoint, spec.StartRotation)
                : ArticulatedRobotRig.SolveInverseKinematicsStatic(baseOrigin, graspPoint, spec.StartRotation);

            // 2. Heft Waypoint: Sharp, deliberate upward lift (+12 cm) + wrist twist to test payload inertia
            Vector3 liftedPoint = graspPoint + new Vector3(0f, heftLiftMeters, 0f);
            Quaternion heftRotation = spec.StartRotation * Quaternion.Euler(heftWristTwistRad * Mathf.Rad2Deg);
            _qHeftPeak = _rig != null
                ? _rig.SolveInverseKinematics6D(liftedPoint, heftRotation)
                : ArticulatedRobotRig.SolveInverseKinematicsStatic(baseOrigin, liftedPoint, heftRotation);

            if (dof >= 6)
            {
                float maxAngle = ArticulatedRobotRig.MaxJointAngleRad;
                _qHeftPeak[3] = Mathf.Clamp(_qHeftPeak[3] + heftWristTwistRad.x, -maxAngle, maxAngle);
                _qHeftPeak[4] = Mathf.Clamp(_qHeftPeak[4] + heftWristTwistRad.y, -maxAngle, maxAngle);
                _qHeftPeak[5] = Mathf.Clamp(_qHeftPeak[5] + heftWristTwistRad.z, -maxAngle, maxAngle);
            }

            // 3. Transport Waypoint: Carry to new 3D location while twisting wrist joints (flat -> upright)
            _qTarget6D = _rig != null
                ? _rig.SolveInverseKinematics6D(spec.TargetPosition, spec.TargetRotation)
                : ArticulatedRobotRig.SolveInverseKinematicsStatic(baseOrigin, spec.TargetPosition, spec.TargetRotation);

            if (dof >= 6)
            {
                float maxAngle = ArticulatedRobotRig.MaxJointAngleRad;
                float sideSign = Mathf.Sign(spec.TargetPosition.x == 0f ? 1f : spec.TargetPosition.x);
                // Deliberate ~80-90 deg wrist twist during transport so the flat payload is rotated upright
                _qTarget6D[3] = Mathf.Clamp(_qTarget6D[3] - 0.55f, -maxAngle, maxAngle);
                _qTarget6D[4] = Mathf.Clamp(_qHeftPeak[4] + sideSign * 0.95f, -maxAngle, maxAngle);
                _qTarget6D[5] = Mathf.Clamp(_qHeftPeak[5] + sideSign * 1.45f, -maxAngle, maxAngle);
            }
        }

        /// <summary>
        /// Evaluates the sequential waypoint PD controller at discrete step t of totalSteps,
        /// updating the phase flag (0 = BlindApproach, 1 = SensoryHeft, 2 = BlindTransport),
        /// triggering the instant rigid FixedJoint lock at the onset of Phase 1.5, and returning u_t in [-1, 1]^6.
        /// </summary>
        public float[] StepWaypointController(int stepIndex, int totalSteps, float dt)
        {
            int dof = _rig.JointCount;
            if (_currentActionVector == null || _currentActionVector.Length != dof)
            {
                InitializeControllerState(seed: 42);
            }

            float[] q = _rig.GetJointPositions();
            float[] p = _rig.GetJointVelocities();
            float[] qDes = new float[dof];
            float[] dqDes = new float[dof];

            int approachEnd = setupHoldSteps + blindApproachSteps;
            int heftEnd = approachEnd + sensoryHeftSteps;
            float gainMultiplier = 1.0f;

            if (stepIndex < setupHoldSteps)
            {
                // Phase 0 (The Neutral Start): Hold clean neutral resting posture
                _currentPhase = GraspTrajectoryPhase.BlindApproach;
                Array.Copy(_qNeutral, qDes, dof);
            }
            else if (stepIndex < approachEnd)
            {
                // Phase 1 (The Approach): Deliberate minimum-jerk sweep to the ground object's top grasp surface
                _currentPhase = GraspTrajectoryPhase.BlindApproach;
                float tau = Mathf.Clamp01((float)(stepIndex - setupHoldSteps + 1) / Mathf.Max(1, blindApproachSteps));
                float s = SmoothStepMinimumJerk(tau);
                float ds = SmoothStepMinimumJerkDerivative(tau) / Mathf.Max(dt * blindApproachSteps, 1e-3f);
                for (int i = 0; i < dof; i++)
                {
                    qDes[i] = Mathf.Lerp(_qNeutral[i], _qGrasp[i], s);
                    dqDes[i] = (_qGrasp[i] - _qNeutral[i]) * ds;
                }
            }
            else if (stepIndex < heftEnd)
            {
                // Phase 1.5 (The "Heft"): Instant rigid lock + sharp, deliberate upward lift to feel the payload's weight
                _currentPhase = GraspTrajectoryPhase.SensoryHeft;
                if (_spawner != null && !_spawner.IsPayloadGrasped)
                {
                    _spawner.EnsureRigidGrasp(_rig);
                }

                float tau = Mathf.Clamp01((float)(stepIndex - approachEnd + 1) / Mathf.Max(1, sensoryHeftSteps));
                // Fast, monotonic upward lift profile (completes lift in first 75% of Heft window and holds)
                float fastTau = Mathf.Clamp01(tau * 1.35f);
                float liftProgress = SmoothStepMinimumJerk(fastTau);
                float dLift = (tau * 1.35f <= 1.0f)
                    ? (SmoothStepMinimumJerkDerivative(fastTau) * 1.35f / Mathf.Max(dt * sensoryHeftSteps, 1e-3f))
                    : 0.0f;
                gainMultiplier = heftAccelerationBoost;

                for (int i = 0; i < dof; i++)
                {
                    float deltaHeft = _qHeftPeak[i] - _qGrasp[i];
                    qDes[i] = _qGrasp[i] + deltaHeft * liftProgress;
                    dqDes[i] = deltaHeft * dLift;
                }
            }
            else
            {
                // Phase 2 (The Transport): Deliberate minimum-jerk carry from lifted pose to new 3D target location
                // while smoothly twisting the wrist joints to rotate the payload from flat to upright.
                _currentPhase = GraspTrajectoryPhase.BlindTransport;
                if (_spawner != null && !_spawner.IsPayloadGrasped)
                {
                    _spawner.EnsureRigidGrasp(_rig);
                }

                int transportSteps = Mathf.Max(1, totalSteps - heftEnd);
                float tau = Mathf.Clamp01((float)(stepIndex - heftEnd + 1) / transportSteps);
                float s = SmoothStepMinimumJerk(tau);
                float ds = SmoothStepMinimumJerkDerivative(tau) / Mathf.Max(dt * transportSteps, 1e-3f);

                for (int i = 0; i < dof; i++)
                {
                    qDes[i] = Mathf.Lerp(_qHeftPeak[i], _qTarget6D[i], s);
                    dqDes[i] = (_qTarget6D[i] - _qHeftPeak[i]) * ds;
                }
            }

            float invScale = 1.0f / Mathf.Max(0.1f, targetAngleScaleRad);
            float[] u = new float[dof];
            for (int i = 0; i < dof; i++)
            {
                float posErr = qDes[i] - q[i];
                float velErr = dqDes[i] - p[i];
                float pdTargetRad = qDes[i] + gainMultiplier * (proportionalGainKp * posErr + derivativeGainKd * velErr);
                u[i] = Mathf.Clamp(pdTargetRad * invScale, -1.0f, 1.0f);
            }

            SetActionVector(u);
            return u;
        }

        private static float SmoothStepMinimumJerk(float tau)
        {
            float t = Mathf.Clamp01(tau);
            return t * t * t * (10.0f + t * (-15.0f + 6.0f * t));
        }

        private static float SmoothStepMinimumJerkDerivative(float tau)
        {
            float t = Mathf.Clamp01(tau);
            return 30.0f * t * t * (1.0f - t) * (1.0f - t);
        }

        /// <summary>
        /// Sets the continuous action vector u_t in [-1, 1]^6 to be applied on the next physics step.
        /// </summary>
        public void SetActionVector(float[] actionVector)
        {
            if (actionVector == null)
            {
                throw new ArgumentNullException(nameof(actionVector));
            }
            if (actionVector.Length != _rig.JointCount)
            {
                throw new ArgumentException(
                    $"Expected action vector of dimension {_rig.JointCount}, got {actionVector.Length}");
            }
            if (_currentActionVector == null || _currentActionVector.Length != actionVector.Length)
            {
                _currentActionVector = new float[actionVector.Length];
            }
            Array.Copy(actionVector, _currentActionVector, actionVector.Length);
        }

        /// <summary>
        /// Applies the continuous 6-DOF motor action vector u_t to all ArticulationBody joints.
        /// </summary>
        public void ApplyActionsToJoints()
        {
            IReadOnlyList<ArticulationBody> joints = _rig.ActuatedJoints;
            float[] q = _rig.GetJointPositions();
            float[] p = _rig.GetJointVelocities();

            for (int i = 0; i < joints.Count; i++)
            {
                ArticulationBody joint = joints[i];
                float u_i = Mathf.Clamp(_currentActionVector[i], -1.0f, 1.0f);

                if (controlMode == ActionControlMode.MotorTargetPD)
                {
                    var drive = joint.xDrive;
                    float targetRad = u_i * targetAngleScaleRad;
                    drive.target = targetRad * Mathf.Rad2Deg;
                    drive.targetVelocity = -viscousJointDamping * p[i] * Mathf.Rad2Deg;
                    joint.xDrive = drive;
                }
                else
                {
                    float netTorque = (u_i * maxMotorTorque) - (viscousJointDamping * maxMotorTorque * p[i]) - (15.0f * q[i]);
                    joint.jointForce = new ArticulationReducedSpace(netTorque);
                }
            }
        }

        private void FixedUpdate()
        {
            if (_rig != null && _rig.JointCount > 0 && _currentActionVector != null)
            {
                ApplyActionsToJoints();
            }
        }
    }
}
