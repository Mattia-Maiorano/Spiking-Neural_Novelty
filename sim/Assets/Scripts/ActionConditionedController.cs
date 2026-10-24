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

    public enum AttractorRegime
    {
        LimitCycleWalkingGait = 0,
        FixedPointBalancing = 1,
        ExternalActionStream = 2
    }

    /// <summary>
    /// Action-Conditioned Movement Controller for Articulated Robotic Entities.
    /// Receives continuous motor action vectors u_t (motor target angles or joint torques)
    /// and applies them to ArticulationBody joints alongside realistic viscous damping,
    /// gravity, and ground-contact friction.
    /// Supports both stable limit-cycle attractors (rhythmic CPG walking gaits) and
    /// fixed-point attractors (postural balancing).
    /// </summary>
    [RequireComponent(typeof(ArticulatedRobotRig))]
    public class ActionConditionedController : MonoBehaviour
    {
        [Header("Action & Physics Parameters")]
        [SerializeField] private ActionControlMode controlMode = ActionControlMode.MotorTargetPD;
        [SerializeField] private AttractorRegime regime = AttractorRegime.LimitCycleWalkingGait;
        [SerializeField] private float viscousJointDamping = 0.25f;
        [SerializeField] private float maxMotorTorque = 160.0f;
        [SerializeField] private float targetAngleScaleRad = 0.85f;
        [SerializeField] private Vector3 gravityVector = new Vector3(0f, -9.81f, 0f);

        [Header("CPG Limit-Cycle Attractor Parameters")]
        [SerializeField] private float gaitFrequencyHz = 1.4f;
        [SerializeField] private float gaitAmplitude = 0.65f;

        private ArticulatedRobotRig _rig;
        private float[] _currentActionVector;
        private float[] _phaseOffsets;
        private float[] _fixedPointEquilibrium;

        public int ActionDim => _rig != null ? _rig.JointCount : 0;
        public float[] CurrentAction => _currentActionVector;
        public AttractorRegime CurrentRegime { get => regime; set => regime = value; }

        private void Awake()
        {
            _rig = GetComponent<ArticulatedRobotRig>();
            Physics.gravity = gravityVector;
            InitializeControllerState(seed: 42);
        }

        public void InitializeControllerState(int seed)
        {
            int dof = _rig.JointCount;
            _currentActionVector = new float[dof];
            _phaseOffsets = new float[dof];
            _fixedPointEquilibrium = new float[dof];

            System.Random rng = new System.Random(seed);
            // Anti-phase bilateral coupling for left/right limbs to produce realistic walking gaits
            for (int i = 0; i < dof; i++)
            {
                bool isRightSide = i >= (dof / 2);
                float basePhase = isRightSide ? Mathf.PI : 0.0f;
                float jointHierarchyPhase = (i % (Mathf.Max(1, dof / 2))) * 0.45f;
                _phaseOffsets[i] = basePhase + jointHierarchyPhase;
                _fixedPointEquilibrium[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
            }
        }

        /// <summary>
        /// Computes a continuous action vector u_t for timestep t based on the active attractor regime.
        /// Even trajectories can execute rhythmic CPG limit cycles while odd trajectories execute fixed-point posture control,
        /// or external actions can be injected directly via SetActionVector(u_t).
        /// </summary>
        public float[] SampleRegimeAction(float timeSeconds, int trajectoryId)
        {
            int dof = _rig.JointCount;
            if (_currentActionVector == null || _currentActionVector.Length != dof)
            {
                InitializeControllerState(trajectoryId);
            }

            float[] u = new float[dof];
            if (regime == AttractorRegime.LimitCycleWalkingGait)
            {
                float omega = 2.0f * Mathf.PI * gaitFrequencyHz;
                for (int i = 0; i < dof; i++)
                {
                    float harmonic1 = Mathf.Sin(omega * timeSeconds + _phaseOffsets[i]);
                    float harmonic2 = 0.25f * Mathf.Cos(2.0f * omega * timeSeconds + _phaseOffsets[i]);
                    u[i] = Mathf.Clamp(gaitAmplitude * (harmonic1 + harmonic2), -1.0f, 1.0f);
                }
            }
            else if (regime == AttractorRegime.FixedPointBalancing)
            {
                float[] q = _rig.GetJointPositions();
                float[] p = _rig.GetJointVelocities();
                for (int i = 0; i < dof; i++)
                {
                    float posError = _fixedPointEquilibrium[i] - q[i];
                    float velDamp = -0.35f * p[i];
                    u[i] = Mathf.Clamp(posError + velDamp, -1.0f, 1.0f);
                }
            }
            else
            {
                Array.Copy(_currentActionVector, u, dof);
            }

            SetActionVector(u);
            return u;
        }

        /// <summary>
        /// Sets the continuous action vector u_t in R^{D_u} to be applied on the next physics step.
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
        /// Applies the continuous action vector u_t to all ArticulationBody joints with dissipative damping.
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
                    // Direct torque actuation with dissipative viscous damping (-c * q_dot) and restorative spring
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
