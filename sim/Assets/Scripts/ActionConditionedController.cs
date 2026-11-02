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
    /// Sequential Waypoint PD Controller for the 6-DOF Industrial Manipulator (SPWM-v0.6.4).
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
        [SerializeField] private float proportionalGainKp = 0.35f;
        [SerializeField] private float derivativeGainKd = 0.08f;
        [SerializeField] private float viscousJointDamping = 0.25f;
        [SerializeField] private float maxMotorTorque = 240.0f;
        [SerializeField] private float targetAngleScaleRad = ArticulatedRobotRig.MaxJointAngleRad;
        [SerializeField] private Vector3 gravityVector = new Vector3(0f, -9.81f, 0f);

        [Header("Phasic Waypoint Timing (Steps)")]
        [SerializeField] private int setupHoldSteps = 6;
        [SerializeField] private int blindApproachSteps = 34;
        [Tooltip("Number of approach steps before contact to enter Phase 1.5 (sensory grounding).")]
        [SerializeField] private int preContactSensorySteps = 5;
        [SerializeField] private int sensoryHeftSteps = 12;

        [Header("Temporal & Controller Randomization (Anti-Memorization)")]
        [Tooltip("If enabled, wildly varies approach duration and PD parameters per episode so the predictor cannot memorize timing.")]
        [SerializeField] private bool enableTemporalRandomization = true;
        [Tooltip("Minimum approach steps (earliest lift onset ~ step 15).")]
        [SerializeField] private int minApproachSteps = 14;
        [Tooltip("Maximum approach steps (latest lift onset ~ step 60).")]
        [SerializeField] private int maxApproachSteps = 58;
        [Tooltip("Minimum proportional gain (stiffness) multiplier during approach.")]
        [SerializeField] private float minApproachKpMult = 0.65f;
        [Tooltip("Maximum proportional gain (stiffness) multiplier during approach.")]
        [SerializeField] private float maxApproachKpMult = 1.45f;
        [Tooltip("Minimum derivative gain (damping) multiplier during approach.")]
        [SerializeField] private float minApproachKdMult = 0.65f;
        [Tooltip("Maximum derivative gain (damping) multiplier during approach.")]
        [SerializeField] private float maxApproachKdMult = 1.45f;

        [Header("Sensory Heft Sharp Upward Lift")]
        [SerializeField] private float heftLiftMeters = 0.10f;
        [SerializeField] private Vector3 heftWristTwistRad = new Vector3(0.08f, -0.05f, -0.20f);

        [Header("Hand Orientation (palm normal polar angle in frontal X-Y plane)")]
        [Tooltip("270 deg = palm facing straight down at grasp.")]
        [SerializeField] private float graspPalmAngleDeg = 270f;
        [Tooltip("135 deg = palm open, facing up and to the left at the final pose.")]
        [SerializeField] private float finalPalmAngleDeg = 135f;

        private ArticulatedRobotRig _rig;
        private ProceduralObjectSpawner _spawner;
        private float[] _currentActionVector;
        private float[] _currentPosTarget;   // reference joint position (rad) fed to the drive
        private float[] _currentVelTarget;   // reference joint velocity (rad/s) fed to the drive
        private float _stiffnessMult = 1f;   // per-phase/episode multiplier on drive stiffness
        private float _dampingMult = 1f;     // per-phase/episode multiplier on drive damping
        private float[] _baseStiffness;
        private float[] _baseDamping;

        // Per-episode randomized approach duration and PD parameters
        private int _activeBlindApproachSteps;
        private float _activeApproachKp;
        private float _activeApproachKd;

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
        public int ActiveBlindApproachSteps => _activeBlindApproachSteps;
        public int ActiveGraspLockStep => setupHoldSteps + _activeBlindApproachSteps;

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
            if (_rig == null)
            {
                _rig = GetComponent<ArticulatedRobotRig>();
            }
            if (_spawner == null)
            {
                _spawner = GetComponent<ProceduralObjectSpawner>();
            }

            int dof = _rig != null && _rig.JointCount > 0 ? _rig.JointCount : ArticulatedRobotRig.CanonicalJointCount;
            _currentActionVector = new float[dof];
            _currentPosTarget = new float[dof];
            _currentVelTarget = new float[dof];
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

            // Compute per-episode temporal timing and PD stiffness/damping parameters
            if (enableTemporalRandomization)
            {
                System.Random rng = new System.Random(seed * 3571 + 65537);
                _activeBlindApproachSteps = rng.Next(minApproachSteps, maxApproachSteps + 1);
                float kpMult = Mathf.Lerp(minApproachKpMult, maxApproachKpMult, (float)rng.NextDouble());
                float kdMult = Mathf.Lerp(minApproachKdMult, maxApproachKdMult, (float)rng.NextDouble());
                _activeApproachKp = proportionalGainKp * kpMult;
                _activeApproachKd = derivativeGainKd * kdMult;
            }
            else
            {
                _activeBlindApproachSteps = blindApproachSteps;
                _activeApproachKp = proportionalGainKp;
                _activeApproachKd = derivativeGainKd;
            }

            // 1. Approach Waypoint: End-effector tip reaches the exact top surface of the ground object
            Vector3 graspPoint = spec.GraspPoint;
            Vector3 graspPalm = ArticulatedRobotRig.PalmNormalFromFrontalAngleDeg(graspPalmAngleDeg);
            Vector3 finalPalm = ArticulatedRobotRig.PalmNormalFromFrontalAngleDeg(finalPalmAngleDeg);
            _qGrasp = _rig != null
                ? _rig.SolveInverseKinematics6DOriented(graspPoint, graspPalm)
                : ArticulatedRobotRig.SolveInverseKinematicsOrientedStatic(baseOrigin, graspPoint, graspPalm);

            // 2. Heft Waypoint: Sharp, deliberate upward lift (+12 cm) + wrist twist to test payload inertia
            Vector3 liftedPoint = graspPoint + new Vector3(0f, heftLiftMeters, 0f);
            _qHeftPeak = _rig != null
                ? _rig.SolveInverseKinematics6DOriented(liftedPoint, graspPalm)
                : ArticulatedRobotRig.SolveInverseKinematicsOrientedStatic(baseOrigin, liftedPoint, graspPalm);

            if (dof >= 6)
            {
                float maxAngle = ArticulatedRobotRig.MaxJointAngleRad;
                // Wrist_Pitch / Wrist_Yaw stay straight; heft twist is applied to Wrist_Roll (pronation/supination) only.
                _qHeftPeak[5] = Mathf.Clamp(_qHeftPeak[5] + heftWristTwistRad.z, -maxAngle, maxAngle);
            }

            // 3. Transport Waypoint: Carry to new 3D location while twisting wrist joints (flat -> upright)
            //    The wrist rotates from palm-down (270 deg) to palm-open up-left (135 deg) and terminates there.
            _qTarget6D = _rig != null
                ? _rig.SolveInverseKinematics6DOriented(spec.TargetPosition, finalPalm)
                : ArticulatedRobotRig.SolveInverseKinematicsOrientedStatic(baseOrigin, spec.TargetPosition, finalPalm);
        }

        /// <summary>
        /// Evaluates the sequential waypoint PD controller at discrete step t of totalSteps,
        /// updating the phase flag (0 = BlindApproach, 1 = SensoryHeft, 2 = BlindTransport),
        /// triggering the instant rigid FixedJoint lock at the onset of Phase 1.5, and returning u_t in [-1, 1]^6.
        /// Temporal approach steps and PD stiffness vary widely across episodes to prevent timing memorization.
        /// </summary>
        public float[] StepWaypointController(int stepIndex, int totalSteps, float dt)
        {
            if (_rig == null)
            {
                _rig = GetComponent<ArticulatedRobotRig>();
            }

            int dof = _rig != null ? _rig.JointCount : ArticulatedRobotRig.CanonicalJointCount;
            if (_currentActionVector == null || _currentActionVector.Length != dof)
            {
                InitializeControllerState(seed: 42);
            }

            float[] q = _rig != null ? _rig.GetJointPositions() : new float[dof];
            float[] p = _rig != null ? _rig.GetJointVelocities() : new float[dof];
            float[] qDes = new float[dof];
            float[] dqDes = new float[dof];

            int curApproachSteps = Mathf.Max(1, _activeBlindApproachSteps > 0 ? _activeBlindApproachSteps : blindApproachSteps);
            int approachEnd = setupHoldSteps + curApproachSteps;
            int heftStart = Mathf.Max(setupHoldSteps + 1, approachEnd - preContactSensorySteps);
            int heftEnd = approachEnd + sensoryHeftSteps;
            float gainMultiplier = 1.0f;
            float curKp = _activeApproachKp > 0f ? _activeApproachKp : proportionalGainKp;
            float curKd = _activeApproachKd > 0f ? _activeApproachKd : derivativeGainKd;

            if (stepIndex < setupHoldSteps)
            {
                // Phase 0 (The Neutral Start): Hold clean neutral resting posture
                _currentPhase = GraspTrajectoryPhase.BlindApproach;
                Array.Copy(_qNeutral, qDes, dof);
            }
            else if (stepIndex < heftStart)
            {
                // Phase 1 (The Blind Approach): Deliberate minimum-jerk sweep toward the ground object
                _currentPhase = GraspTrajectoryPhase.BlindApproach;
                float tau = Mathf.Clamp01((float)(stepIndex - setupHoldSteps + 1) / curApproachSteps);
                float s = SmoothStepMinimumJerk(tau);
                float ds = SmoothStepMinimumJerkDerivative(tau) / Mathf.Max(dt * curApproachSteps, 1e-3f);
                for (int i = 0; i < dof; i++)
                {
                    qDes[i] = Mathf.Lerp(_qNeutral[i], _qGrasp[i], s);
                    dqDes[i] = (_qGrasp[i] - _qNeutral[i]) * ds;
                }
            }
            else if (stepIndex < approachEnd)
            {
                // Phase 1.5 Pre-Contact (Sensory Grounding Approach):
                // Enter Phase 1.5 n steps before contact to inject ground truth before reaching the object
                _currentPhase = GraspTrajectoryPhase.SensoryHeft;
                float tau = Mathf.Clamp01((float)(stepIndex - setupHoldSteps + 1) / curApproachSteps);
                float s = SmoothStepMinimumJerk(tau);
                float ds = SmoothStepMinimumJerkDerivative(tau) / Mathf.Max(dt * curApproachSteps, 1e-3f);
                for (int i = 0; i < dof; i++)
                {
                    qDes[i] = Mathf.Lerp(_qNeutral[i], _qGrasp[i], s);
                    dqDes[i] = (_qGrasp[i] - _qNeutral[i]) * ds;
                }
            }
            else if (stepIndex < heftEnd)
            {
                // Phase 1.5 Post-Contact (The "Heft" Lift):
                // Instant rigid lock at contact (stepIndex == approachEnd) + smooth upward lift
                _currentPhase = GraspTrajectoryPhase.SensoryHeft;
                if (_spawner != null && !_spawner.IsPayloadGrasped)
                {
                    _spawner.EnsureRigidGrasp(_rig);
                }

                float tau = Mathf.Clamp01((float)(stepIndex - approachEnd) / Mathf.Max(1, sensoryHeftSteps));
                // Softened minimum-jerk upward lift profile (smooth across full heft window without harsh 1.35x compression)
                float liftProgress = SmoothStepMinimumJerk(tau);
                float dLift = SmoothStepMinimumJerkDerivative(tau) / Mathf.Max(dt * sensoryHeftSteps, 1e-3f);
                curKp = proportionalGainKp;
                curKd = derivativeGainKd;

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
                curKp = proportionalGainKp;
                curKd = derivativeGainKd;

                for (int i = 0; i < dof; i++)
                {
                    qDes[i] = Mathf.Lerp(_qHeftPeak[i], _qTarget6D[i], s);
                    dqDes[i] = (_qTarget6D[i] - _qHeftPeak[i]) * ds;
                }
            }

            // The ArticulationBody drive IS the PD controller: command the reference directly
            // (no outer PD on top, which previously produced loop gain < -1 and a Nyquist oscillation).
            // Kp/Kd randomization is applied to the drive's stiffness/damping instead.
            _stiffnessMult = gainMultiplier * curKp / Mathf.Max(1e-6f, proportionalGainKp);
            _dampingMult = curKd / Mathf.Max(1e-6f, derivativeGainKd);

            if (_baseStiffness == null || _baseStiffness.Length != dof)
            {
                _baseStiffness = new float[dof];
                _baseDamping = new float[dof];
                if (_rig != null && _rig.ActuatedJoints != null)
                {
                    var joints = _rig.ActuatedJoints;
                    for (int i = 0; i < Mathf.Min(dof, joints.Count); i++)
                    {
                        _baseStiffness[i] = joints[i].xDrive.stiffness;
                        _baseDamping[i] = joints[i].xDrive.damping;
                    }
                }
            }

            if (_currentPosTarget == null || _currentPosTarget.Length != dof)
            {
                _currentPosTarget = new float[dof];
            }
            if (_currentVelTarget == null || _currentVelTarget.Length != dof)
            {
                _currentVelTarget = new float[dof];
            }

            float invScale = 1.0f / Mathf.Max(0.1f, targetAngleScaleRad);
            float[] u = new float[dof];
            for (int i = 0; i < dof; i++)
            {
                _currentPosTarget[i] = qDes[i];
                _currentVelTarget[i] = dqDes[i];
                u[i] = Mathf.Clamp(qDes[i] * invScale, -1.0f, 1.0f);
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
            if (_rig == null)
            {
                _rig = GetComponent<ArticulatedRobotRig>();
            }
            if (_rig == null) return;

            IReadOnlyList<ArticulationBody> joints = _rig.ActuatedJoints;
            float[] q = _rig.GetJointPositions();
            float[] p = _rig.GetJointVelocities();

            if (_baseStiffness == null || _baseStiffness.Length != joints.Count)
            {
                _baseStiffness = new float[joints.Count];
                _baseDamping = new float[joints.Count];
                for (int i = 0; i < joints.Count; i++)
                {
                    _baseStiffness[i] = joints[i].xDrive.stiffness;
                    _baseDamping[i] = joints[i].xDrive.damping;
                }
            }
            if (_currentVelTarget == null || _currentVelTarget.Length != joints.Count)
            {
                _currentVelTarget = new float[joints.Count];
            }

            for (int i = 0; i < joints.Count; i++)
            {
                ArticulationBody joint = joints[i];
                float u_i = Mathf.Clamp(_currentActionVector[i], -1.0f, 1.0f);

                if (controlMode == ActionControlMode.MotorTargetPD)
                {
                    var drive = joint.xDrive;
                    float targetRad = (_currentPosTarget != null && i < _currentPosTarget.Length)
                        ? _currentPosTarget[i]
                        : (u_i * targetAngleScaleRad);
                    drive.target = targetRad * Mathf.Rad2Deg;
                    drive.targetVelocity = _currentVelTarget[i] * Mathf.Rad2Deg;
                    drive.stiffness = _baseStiffness[i] * _stiffnessMult;
                    drive.damping = _baseDamping[i] * _dampingMult;
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
