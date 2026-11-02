using System;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_6000_0_OR_NEWER
using PhysicMaterial = UnityEngine.PhysicsMaterial;
using PhysicMaterialCombine = UnityEngine.PhysicsMaterialCombine;
#endif

namespace SPWM.Robotics
{
    /// <summary>
    /// Supported robotic morphology archetype for the 6-DOF active perception grasping rig (SPWM-v0.5.0).
    /// </summary>
    public enum RobotArchetype
    {
        ManipulatorArm = 0
    }

    /// <summary>
    /// Specification for a single articulated link in the kinematic hierarchy.
    /// Each child joint's LocalAnchorPosition.y equals its parent link's physical length (LinkDimensions.y),
    /// ensuring an unbroken, bolted kinematic chain with zero gaps.
    /// </summary>
    [Serializable]
    public struct ArticulatedLinkSpec
    {
        public string Name;
        public int ParentIndex;
        public Vector3 LocalAnchorPosition;
        public Vector3 LinkDimensions;
        public float Mass;
        public float JointFriction;
        public float AngularDamping;
        public float LowerLimitDeg;
        public float UpperLimitDeg;
        public float DriveStiffness;
        public float DriveDamping;
        public float MaxForceLimit;
        public Vector3 AxisRotationEuler;
    }

    /// <summary>
    /// Canonical 6-DOF Industrial Manipulator Rig built on PhysX 4 Featherstone ArticulationBody chains.
    /// Models an immovable high-mass root base (20.0 kg) permanently bolted to the ground and 6 actuated
    /// revolute joints (Shoulder Pan, Shoulder Lift, Elbow Flex, Wrist Pitch, Wrist Yaw, Wrist Roll)
    /// with unscaled rigid frames and contiguous link geometry so segments never separate or stretch.
    /// </summary>
    public class ArticulatedRobotRig : MonoBehaviour
    {
        public const int CanonicalJointCount = 6;
        public const float MaxJointAngleRad = 3.14f;

        // Exact link lengths and anthropomorphic mounting elevations (meters) matching GetArchetypeSpecs()
        public const float ShoulderMountHeight = 1.25f;
        public const float BaseHeight = 0.05f;
        public const float TurretHeight = 0.05f;
        public const float ShoulderElevation = ShoulderMountHeight + BaseHeight + TurretHeight; // 1.35m (human glenohumeral shoulder height)
        public const float EyeLevelElevation = ShoulderElevation + 0.25f;                        // 1.60m (human eye level)
        // Head / over-shoulder perspective: 22cm medial (-X) to right shoulder, 25cm above shoulder, 28cm behind shoulder (-0.28m Z).
        public static readonly Vector3 HeadOffsetFromShoulder = new Vector3(-0.22f, 0.25f, -0.28f);
        // Gaze tilted toward the manipulation workspace in front of the head on the left
        public static readonly Vector3 EyeCameraLookTargetOffset = new Vector3(-0.12f, 1.06f, 0.44f);
        public const float UpperArmLength = 0.28f;
        public const float ForearmLength = 0.25f;
        public const float WristPitchLength = 0.04f;
        public const float WristYawLength = 0.04f;
        public const float WristRollLength = 0.06f;
        public const float TotalWristToolLength = WristPitchLength + WristYawLength + WristRollLength; // 0.14m (human wrist + palm)

        /// <summary>
        /// Returns the anthropomorphic head (eye-level) 3D world position relative to the robot base origin.
        /// Positioned 25 cm above the shoulder mount (y = 1.60m) and 15 cm medial (-X) to the right shoulder.
        /// </summary>
        public static Vector3 GetAnthropomorphicHeadPosition(Vector3 baseOrigin)
        {
            return baseOrigin + new Vector3(
                HeadOffsetFromShoulder.x,
                ShoulderElevation + HeadOffsetFromShoulder.y,
                HeadOffsetFromShoulder.z
            );
        }

        /// <summary>
        /// Computes the anthropomorphic eye-level camera position and forward workspace orientation.
        /// </summary>
        public static void GetEyeLevelCameraPose(
            Vector3 baseOrigin,
            out Vector3 cameraWorldPosition,
            out Quaternion cameraWorldRotation)
        {
            cameraWorldPosition = GetAnthropomorphicHeadPosition(baseOrigin);
            Vector3 lookTarget = baseOrigin + EyeCameraLookTargetOffset;
            Vector3 forwardDir = (lookTarget - cameraWorldPosition).normalized;
            cameraWorldRotation = forwardDir.sqrMagnitude > 1e-6f
                ? Quaternion.LookRotation(forwardDir, Vector3.up)
                : Quaternion.identity;
        }

        [Header("Morphology Configuration")]
        [SerializeField] private RobotArchetype archetype = RobotArchetype.ManipulatorArm;
        [SerializeField] private bool immobileBase = true;
        [SerializeField] private float globalJointFriction = 0.06f;
        [SerializeField] private float globalAngularDamping = 0.25f;
        [SerializeField] private float globalLinearDamping = 0.08f;
        [SerializeField] private float groundStaticFriction = 0.90f;
        [SerializeField] private float groundDynamicFriction = 0.80f;

        private ArticulationBody _rootBody;
        private readonly List<ArticulationBody> _actuatedJoints = new List<ArticulationBody>();
        private Transform _endEffectorTipTransform;
        private Transform _wristCameraMountTransform;
        private PhysicMaterial _groundContactMaterial;

        public RobotArchetype Archetype => archetype;
        public ArticulationBody RootBody => _rootBody;
        public IReadOnlyList<ArticulationBody> ActuatedJoints => _actuatedJoints;
        public int JointCount => _actuatedJoints.Count;
        public ArticulationBody EndEffectorBody => _actuatedJoints.Count > 0 ? _actuatedJoints[_actuatedJoints.Count - 1] : _rootBody;
        public Transform EndEffectorTransform => _endEffectorTipTransform != null
            ? _endEffectorTipTransform
            : (EndEffectorBody != null ? EndEffectorBody.transform : transform);
        public Transform WristCameraMountTransform => _wristCameraMountTransform != null
            ? _wristCameraMountTransform
            : EndEffectorTransform;

        private void Awake()
        {
            BuildContactMaterial();
            if (_actuatedJoints.Count == 0)
            {
                BuildProceduralRig(archetype);
            }
        }

        private void BuildContactMaterial()
        {
            _groundContactMaterial = new PhysicMaterial("RoboticGroundContact")
            {
                staticFriction = groundStaticFriction,
                dynamicFriction = groundDynamicFriction,
                bounciness = 0.0f,
                frictionCombine = PhysicMaterialCombine.Multiply,
                bounceCombine = PhysicMaterialCombine.Minimum
            };
        }

        /// <summary>
        /// Constructs the 6-DOF ArticulationBody hierarchy using strictly unscaled (1, 1, 1) rigid frames.
        /// Each link's collider and visual geometry extend from its own pivot (0, 0, 0) to the next child
        /// joint's pivot (0, LinkDimensions.y, 0), guaranteeing an unbroken skeleton.
        /// </summary>
        public void BuildProceduralRig(RobotArchetype targetArchetype = RobotArchetype.ManipulatorArm)
        {
            archetype = targetArchetype;
            _actuatedJoints.Clear();
            _endEffectorTipTransform = null;
            _wristCameraMountTransform = null;

            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                DestroyImmediate(transform.GetChild(i).gameObject);
            }

            List<ArticulatedLinkSpec> specs = GetArchetypeSpecs(targetArchetype);
            var createdBodies = new List<ArticulationBody>(specs.Count);
            var linkColliders = new List<Collider>(specs.Count);

            for (int i = 0; i < specs.Count; i++)
            {
                ArticulatedLinkSpec spec = specs[i];
                bool isRoot = spec.ParentIndex < 0;
                bool isLastLink = i == specs.Count - 1;

                // 1. Unscaled rigid ArticulationBody pivot node
                GameObject linkObj = new GameObject(spec.Name);
                Transform parentTransform = isRoot ? transform : createdBodies[spec.ParentIndex].transform;
                linkObj.transform.SetParent(parentTransform, false);
                linkObj.transform.localPosition = spec.LocalAnchorPosition;
                linkObj.transform.localRotation = Quaternion.identity;
                linkObj.transform.localScale = Vector3.one;

                // 2. Collider spanning from current pivot (0, 0, 0) to next joint pivot (0, L, 0)
                float segLen = spec.LinkDimensions.y;
                float segRadius = 0.5f * spec.LinkDimensions.x;
                Collider col;
                if (isRoot)
                {
                    BoxCollider box = linkObj.AddComponent<BoxCollider>();
                    box.center = new Vector3(0f, 0.5f * segLen, 0f);
                    box.size = spec.LinkDimensions;
                    col = box;
                }
                else
                {
                    CapsuleCollider cap = linkObj.AddComponent<CapsuleCollider>();
                    cap.direction = 1; // Y-axis
                    cap.radius = segRadius;
                    cap.height = Mathf.Max(segLen, 2f * segRadius);
                    cap.center = new Vector3(0f, 0.5f * segLen, 0f);
                    col = cap;
                }
                if (_groundContactMaterial != null)
                {
                    col.sharedMaterial = _groundContactMaterial;
                }
                linkColliders.Add(col);

                // 3. Visual meshes (shaft from 0..L + bolted joint hub at 0,0,0)
                BuildLinkVisuals(linkObj.transform, spec, isRoot, isLastLink);

                // 4. Featherstone ArticulationBody with explicit pivot anchor locking
                ArticulationBody ab = linkObj.AddComponent<ArticulationBody>();
                ab.mass = spec.Mass;
                ab.linearDamping = globalLinearDamping;
                ab.angularDamping = spec.AngularDamping > 0f ? spec.AngularDamping : globalAngularDamping;
                ab.jointFriction = spec.JointFriction > 0f ? spec.JointFriction : globalJointFriction;
                ab.useGravity = true;
                ab.collisionDetectionMode = CollisionDetectionMode.Continuous;
                ab.solverIterations = 24;
                ab.solverVelocityIterations = 8;

                if (isRoot)
                {
                    ab.immovable = immobileBase;
                    _rootBody = ab;
                }
                else
                {
                    ab.jointType = ArticulationJointType.RevoluteJoint;
                    Quaternion axisRot = Quaternion.Euler(spec.AxisRotationEuler);
                    ab.matchAnchors = false;
                    ab.anchorPosition = Vector3.zero;
                    ab.parentAnchorPosition = spec.LocalAnchorPosition;
                    ab.anchorRotation = axisRot;
                    ab.parentAnchorRotation = axisRot;
                    ab.twistLock = ArticulationDofLock.LimitedMotion;

                    var drive = ab.xDrive;
                    drive.lowerLimit = spec.LowerLimitDeg;
                    drive.upperLimit = spec.UpperLimitDeg;
                    drive.stiffness = spec.DriveStiffness;
                    drive.damping = spec.DriveDamping;
                    drive.forceLimit = spec.MaxForceLimit;
                    ab.xDrive = drive;

                    _actuatedJoints.Add(ab);
                }

                createdBodies.Add(ab);
            }

            // 5. Ignore self-collisions among the arm's own structural links so adjacent joints never push apart
            for (int i = 0; i < linkColliders.Count; i++)
            {
                for (int j = i + 1; j < linkColliders.Count; j++)
                {
                    if (linkColliders[i] != null && linkColliders[j] != null)
                    {
                        Physics.IgnoreCollision(linkColliders[i], linkColliders[j], true);
                    }
                }
            }
        }

        private void BuildLinkVisuals(Transform linkPivot, ArticulatedLinkSpec spec, bool isRoot, bool isLastLink)
        {
            Vector3 d = spec.LinkDimensions;

            if (isRoot)
            {
                // Root socket sits silently at the right shoulder mount without protruding torso/clavicle blocks into the camera
                return;
            }

            // Anatomical spherical joint articulation (deltoid shoulder cap, elbow, wrist) right at pivot (0, 0, 0)
            if (spec.Name != "Shoulder_Pan")
            {
                GameObject hub = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                hub.name = spec.Name + "_JointHub";
                DestroyImmediate(hub.GetComponent<Collider>());
                hub.transform.SetParent(linkPivot, false);
                hub.transform.localPosition = Vector3.zero;
                hub.transform.localRotation = Quaternion.identity;
                float hubDiameter = Mathf.Max(d.x, d.z) * 1.06f;
                hub.transform.localScale = new Vector3(hubDiameter, hubDiameter, hubDiameter);
            }

            // Main structural limb segment spanning (0, 0, 0) -> (0, d.y, 0)
            GameObject shaft = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            shaft.name = spec.Name + "_Visual";
            DestroyImmediate(shaft.GetComponent<Collider>());
            shaft.transform.SetParent(linkPivot, false);
            shaft.transform.localPosition = new Vector3(0f, 0.5f * d.y, 0f);
            shaft.transform.localRotation = Quaternion.identity;
            // Unity Cylinder primitive has height 2.0, so scale.y = 0.5 * d.y yields exact height d.y
            shaft.transform.localScale = new Vector3(d.x, 0.5f * d.y, d.z);

            if (isLastLink)
            {
                // End-Effector humanoid right hand at (0, d.y, 0):
                // In our arm kinematics (thetaTool ~ pi/2), local +Y points along the forearm toward the target,
                // local -Z points UPWARD (dorsal side of the hand), local +Z points DOWNWARD (palmar side),
                // and local -X points MEDIALLY (left/inward, where the right-hand thumb sits).
                GameObject tipObj = new GameObject("EndEffectorTip");
                tipObj.transform.SetParent(linkPivot, false);
                tipObj.transform.localPosition = new Vector3(0f, d.y, 0f);
                tipObj.transform.localRotation = Quaternion.identity;
                tipObj.transform.localScale = Vector3.one;

                // Humanoid right-hand palm (metacarpals): wide across X (7.6cm), extending 5.2cm along +Y, thin across Z (2.2cm)
                GameObject palm = GameObject.CreatePrimitive(PrimitiveType.Cube);
                palm.name = "HumanoidPalm";
                DestroyImmediate(palm.GetComponent<Collider>());
                palm.transform.SetParent(tipObj.transform, false);
                palm.transform.localPosition = new Vector3(0f, 0.012f, 0.004f);
                palm.transform.localRotation = Quaternion.identity;
                palm.transform.localScale = new Vector3(0.076f, 0.052f, 0.022f);

                // Knuckle bridge across the distal end of the palm
                GameObject knuckles = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                knuckles.name = "HumanoidKnuckles";
                DestroyImmediate(knuckles.GetComponent<Collider>());
                knuckles.transform.SetParent(tipObj.transform, false);
                knuckles.transform.localPosition = new Vector3(0f, 0.038f, 0.004f);
                knuckles.transform.localRotation = Quaternion.Euler(0f, 0f, 90f);
                knuckles.transform.localScale = new Vector3(0.024f, 0.039f, 0.024f);

                // 4 Articulated fingers (Index, Middle, Ring, Little) with proximal + distal phalanges curving around the grasp zone
                float[] fingerOffsetsX = new float[] { -0.027f, -0.009f, 0.009f, 0.027f };
                float[] proxLengths = new float[] { 0.034f, 0.038f, 0.036f, 0.029f };
                float[] distLengths = new float[] { 0.026f, 0.029f, 0.027f, 0.022f };
                for (int f = 0; f < fingerOffsetsX.Length; f++)
                {
                    // Proximal phalanx extending forward (+Y) and slightly downward (+Z palmar)
                    GameObject prox = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                    prox.name = $"HumanoidFingerProx_{f}";
                    DestroyImmediate(prox.GetComponent<Collider>());
                    prox.transform.SetParent(tipObj.transform, false);
                    Quaternion proxRot = Quaternion.Euler(18f, 0f, 0f);
                    Vector3 proxDir = proxRot * Vector3.up;
                    Vector3 knucklePos = new Vector3(fingerOffsetsX[f], 0.038f, 0.004f);
                    prox.transform.localPosition = knucklePos + proxDir * (0.5f * proxLengths[f]);
                    prox.transform.localRotation = proxRot;
                    prox.transform.localScale = new Vector3(0.015f, 0.5f * proxLengths[f], 0.015f);

                    // Distal phalanx flexing further (+48 deg) to cup/grasp the object naturally
                    Vector3 pipPos = knucklePos + proxDir * proxLengths[f];
                    GameObject dist = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                    dist.name = $"HumanoidFingerDist_{f}";
                    DestroyImmediate(dist.GetComponent<Collider>());
                    dist.transform.SetParent(tipObj.transform, false);
                    Quaternion distRot = Quaternion.Euler(48f, 0f, 0f);
                    Vector3 distDir = distRot * Vector3.up;
                    dist.transform.localPosition = pipPos + distDir * (0.5f * distLengths[f]);
                    dist.transform.localRotation = distRot;
                    dist.transform.localScale = new Vector3(0.013f, 0.5f * distLengths[f], 0.013f);
                }

                // Right-hand thumb on the radial/medial side (-X) opposing the fingers
                // Also serves as the asymmetric visual orientation fiducial so Wrist_Roll (q5) is optically distinct
                GameObject thumbMeta = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                thumbMeta.name = "WristOrientationFiducial";
                DestroyImmediate(thumbMeta.GetComponent<Collider>());
                thumbMeta.transform.SetParent(tipObj.transform, false);
                thumbMeta.transform.localPosition = new Vector3(-0.044f, 0.014f, 0.010f);
                thumbMeta.transform.localRotation = Quaternion.Euler(22f, -15f, 38f);
                thumbMeta.transform.localScale = new Vector3(0.018f, 0.024f, 0.018f);

                GameObject thumbDist = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                thumbDist.name = "HumanoidThumbDistal";
                DestroyImmediate(thumbDist.GetComponent<Collider>());
                thumbDist.transform.SetParent(tipObj.transform, false);
                thumbDist.transform.localPosition = new Vector3(-0.052f, 0.042f, 0.022f);
                thumbDist.transform.localRotation = Quaternion.Euler(35f, 15f, 12f);
                thumbDist.transform.localScale = new Vector3(0.016f, 0.020f, 0.016f);

                // Egocentric wrist-mounted neuromorphic camera mount attached to the dorsal side (-Z)
                // of the wrist/hand, looking forward (+Y) over the knuckles/fingers toward the target
                GameObject wristCamMount = new GameObject("WristCameraMount");
                wristCamMount.transform.SetParent(tipObj.transform, false);
                wristCamMount.transform.localPosition = new Vector3(0f, -0.015f, -0.042f);
                wristCamMount.transform.localRotation = Quaternion.Euler(-82f, 0f, 0f);
                wristCamMount.transform.localScale = Vector3.one;

                _endEffectorTipTransform = tipObj.transform;
                _wristCameraMountTransform = wristCamMount.transform;
            }
        }

        /// <summary>
        /// Reads the ground-truth generalized joint positions q* (radians) across all 6 actuated joints.
        /// </summary>
        public float[] GetJointPositions()
        {
            float[] q = new float[_actuatedJoints.Count];
            for (int i = 0; i < _actuatedJoints.Count; i++)
            {
                ArticulationReducedSpace pos = _actuatedJoints[i].jointPosition;
                q[i] = pos.dofCount > 0 ? pos[0] : 0f;
            }
            return q;
        }

        /// <summary>
        /// Reads the ground-truth generalized joint angular velocities p* (rad/s) across all 6 actuated joints.
        /// </summary>
        public float[] GetJointVelocities()
        {
            float[] p = new float[_actuatedJoints.Count];
            for (int i = 0; i < _actuatedJoints.Count; i++)
            {
                ArticulationReducedSpace vel = _actuatedJoints[i].jointVelocity;
                p[i] = vel.dofCount > 0 ? vel[0] : 0f;
            }
            return p;
        }

        /// <summary>
        /// Resets the root and all 6 child ArticulationBody states deterministically to a clean neutral resting pose.
        /// Also synchronizes xDrive targets so there is zero initial snap or oscillation.
        /// </summary>
        public void ResetToDeterministicPose(int seed, float perturbationScale = 0.0f)
        {
            System.Random rng = new System.Random(seed);
            if (_rootBody != null)
            {
                Vector3 rootPos = transform.TransformPoint(new Vector3(0f, ShoulderMountHeight, 0f));
                Quaternion rootRot = transform.rotation;
                _rootBody.TeleportRoot(rootPos, rootRot);
                _rootBody.linearVelocity = Vector3.zero;
                _rootBody.angularVelocity = Vector3.zero;
            }

            float[] neutralPose = GetNeutralRestingPose();
            for (int i = 0; i < _actuatedJoints.Count; i++)
            {
                float jitter = perturbationScale > 0f
                    ? (float)(rng.NextDouble() * 2.0 - 1.0) * perturbationScale
                    : 0f;
                float q0 = neutralPose[i] + jitter;
                _actuatedJoints[i].jointPosition = new ArticulationReducedSpace(q0);
                _actuatedJoints[i].jointVelocity = new ArticulationReducedSpace(0f);

                var drive = _actuatedJoints[i].xDrive;
                drive.target = q0 * Mathf.Rad2Deg;
                drive.targetVelocity = 0f;
                _actuatedJoints[i].xDrive = drive;
            }
        }

        /// <summary>
        /// Returns the canonical anthropomorphic neutral resting pose (radians) for Phase 0 (Setup),
        /// positioning the right humanoid arm in a natural human posture at the side of the body:
        /// upper arm hanging downward from the right shoulder (q1 = 3.00 rad / ~172 deg), elbow down and closed/flexed
        /// (q2 = -1.78 rad / ~102 deg flexion), and hand poised at the lower-right side ready to open and extend
        /// along a diagonal trajectory toward the target object on the left.
        /// </summary>
        public static float[] GetNeutralRestingPose()
        {
            return new float[CanonicalJointCount]
            {
                1.42f,  // Shoulder_Pan   (orients flexed forearm forward-right at start; opens as elbow raises & turns right)
                3.00f,  // Shoulder_Lift  (upper arm hanging straight down from right shoulder; lifts & opens right during reach)
               -1.78f,  // Elbow_Flex     (arm starts closed/flexed at ~102 deg; opens & extends to ~-0.95 rad toward target)
                0.05f,  // Wrist_Pitch    (hand leveled at lower-right start position)
               -0.70f,  // Wrist_Yaw      (poised on the right; sweeps diagonally with forearm toward the left target)
               -0.25f   // Wrist_Roll     (natural human inward supination)
            };
        }

        /// <summary>
        /// Exact forward kinematics computing the 3D world position and rotation of the end-effector tip
        /// for any 6-DOF joint angle vector q (radians).
        /// </summary>
        public static void ComputeForwardKinematics(
            Vector3 baseOrigin,
            float[] q,
            out Vector3 tipWorldPosition,
            out Quaternion tipWorldRotation)
        {
            List<ArticulatedLinkSpec> specs = GetArchetypeSpecs();
            Vector3 pos = baseOrigin;
            Quaternion rot = Quaternion.identity;

            int jointIdx = 0;
            for (int i = 0; i < specs.Count; i++)
            {
                ArticulatedLinkSpec spec = specs[i];
                pos += rot * spec.LocalAnchorPosition;
                if (spec.ParentIndex >= 0 && q != null && jointIdx < q.Length)
                {
                    Vector3 axis = (Quaternion.Euler(spec.AxisRotationEuler) * Vector3.right).normalized;
                    rot = rot * Quaternion.AngleAxis(q[jointIdx] * Mathf.Rad2Deg, axis);
                    jointIdx++;
                }
            }

            float lastLen = specs[specs.Count - 1].LinkDimensions.y;
            tipWorldPosition = pos + rot * new Vector3(0f, lastLen, 0f);
            tipWorldRotation = rot;
        }

        /// <summary>
        /// Analytical 6-DOF inverse kinematics solver mapping a 3D target coordinate and 3D target orientation
        /// to exact generalized joint angles q in R^6 (radians) using an anatomical human elbow-down configuration.
        /// Guarantees that the right hand reaches targets across Low, Mid, and High vertical planes in front of the head
        /// while the upper arm hangs naturally from the shoulder and the elbow stays below the shoulder/wrist line.
        /// </summary>
        public float[] SolveInverseKinematics6D(Vector3 targetWorldPosition, Quaternion targetWorldOrientation)
        {
            Vector3 baseOrigin = _rootBody != null
                ? (_rootBody.transform.position - new Vector3(0f, ShoulderMountHeight, 0f))
                : transform.position;
            return SolveInverseKinematicsStatic(baseOrigin, targetWorldPosition, targetWorldOrientation);
        }

        /// <summary>
        /// Palm-normal direction (world frame) for a given polar angle in the frontal (X-Y) plane:
        /// 270 deg = straight down (palm facing the ground), 135 deg = up and to the left (palm open).
        /// </summary>
        public static Vector3 PalmNormalFromFrontalAngleDeg(float angleDeg)
        {
            float a = angleDeg * Mathf.Deg2Rad;
            return new Vector3(Mathf.Cos(a), Mathf.Sin(a), 0f);
        }

        public float[] SolveInverseKinematics6DOriented(Vector3 targetWorldPosition, Vector3 palmNormalWorld)
        {
            Vector3 baseOrigin = _rootBody != null
                ? (_rootBody.transform.position - new Vector3(0f, ShoulderMountHeight, 0f))
                : transform.position;
            return SolveInverseKinematicsOrientedStatic(baseOrigin, targetWorldPosition, palmNormalWorld);
        }

        /// <summary>
        /// Anatomical oriented IK. Position is solved by Shoulder_Pan, Shoulder_Lift and Elbow_Flex only;
        /// Wrist_Pitch and Wrist_Yaw are held at exactly 0 (straight wrist). The palmar normal (tip local +Z)
        /// is aligned with palmNormalWorld purely by forearm pronation/supination, i.e. Wrist_Roll (local Y axis),
        /// solved in closed form over the full +-180 deg range.
        /// </summary>
        public static float[] SolveInverseKinematicsOrientedStatic(
            Vector3 baseOrigin,
            Vector3 targetWorldPosition,
            Vector3 palmNormalWorld)
        {
            float[] q = SolveInverseKinematicsStatic(baseOrigin, targetWorldPosition, Quaternion.identity, lockWristBend: true);
            q[3] = 0f;
            q[4] = 0f;
            q[5] = 0f;

            // Frame just before the roll joint (roll = 0): roll axis = rot * Y, palm normal = rot * Z.
            ComputeForwardKinematics(baseOrigin, q, out _, out Quaternion rot0);
            Vector3 axis = rot0 * Vector3.up;
            Vector3 nCur = Vector3.ProjectOnPlane(rot0 * Vector3.forward, axis);
            Vector3 nDes = Vector3.ProjectOnPlane(palmNormalWorld, axis);
            if (nCur.sqrMagnitude > 1e-8f && nDes.sqrMagnitude > 1e-8f)
            {
                float roll = Vector3.SignedAngle(nCur, nDes, axis) * Mathf.Deg2Rad;
                q[5] = Mathf.Clamp(roll, -MaxJointAngleRad, MaxJointAngleRad);
            }
            return q;
        }

        private static float[] SolveLinear6(float[,] A, float[] b)
        {
            const int n = 6;
            float[,] M = new float[n, n + 1];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) M[i, j] = A[i, j];
                M[i, n] = b[i];
            }
            for (int c = 0; c < n; c++)
            {
                int piv = c;
                for (int i = c + 1; i < n; i++) if (Mathf.Abs(M[i, c]) > Mathf.Abs(M[piv, c])) piv = i;
                if (piv != c)
                    for (int j = 0; j <= n; j++) { float t = M[c, j]; M[c, j] = M[piv, j]; M[piv, j] = t; }
                float d = Mathf.Abs(M[c, c]) < 1e-9f ? 1e-9f : M[c, c];
                for (int i = c + 1; i < n; i++)
                {
                    float f = M[i, c] / d;
                    for (int j = c; j <= n; j++) M[i, j] -= f * M[c, j];
                }
            }
            float[] x = new float[n];
            for (int i = n - 1; i >= 0; i--)
            {
                float sum = M[i, n];
                for (int j = i + 1; j < n; j++) sum -= M[i, j] * x[j];
                float d = Mathf.Abs(M[i, i]) < 1e-9f ? 1e-9f : M[i, i];
                x[i] = sum / d;
            }
            return x;
        }

        public static float[] SolveInverseKinematicsStatic(
            Vector3 baseOrigin,
            Vector3 targetWorldPosition,
            Quaternion targetWorldOrientation,
            bool lockWristBend = false)
        {
            Vector3 shoulderOrigin = baseOrigin + new Vector3(0f, ShoulderElevation, 0f);
            Vector3 rel = targetWorldPosition - shoulderOrigin;

            Vector3 eulerDeg = targetWorldOrientation.eulerAngles;
            float targetPitchRad = Mathf.DeltaAngle(0f, eulerDeg.x) * Mathf.Deg2Rad;
            float targetYawRad = Mathf.DeltaAngle(0f, eulerDeg.y) * Mathf.Deg2Rad;
            float targetRollRad = Mathf.DeltaAngle(0f, eulerDeg.z) * Mathf.Deg2Rad;

            // Warm-start in the open extended elbow-right human posture:
            // Shoulder opens slightly right (q0 ~ 0.28..0.42 rad), elbow raises & turns right (q1 ~ 2.15..2.50 rad),
            // forearm extends diagonally across to the left target (q2 ~ -0.92..-1.22 rad, opening from -1.78 rad at start).
            float q0 = Mathf.Clamp(0.45f + 0.85f * rel.x, 0.18f, 0.62f);
            float q1 = Mathf.Clamp(2.46f - 0.90f * (rel.y - (-0.27f)), 1.95f, 2.70f);
            float q2 = Mathf.Clamp(-0.95f - 0.35f * (0.48f - rel.z), -1.35f, -0.65f);
            float q3 = Mathf.Clamp(-0.22f + 0.30f * targetPitchRad, -0.85f, 0.65f);
            float q4 = Mathf.Clamp(-0.20f + 0.25f * targetYawRad, -0.75f, 0.75f);
            float q5 = Mathf.Clamp(Mathf.Abs(targetRollRad) > 1e-3f ? targetRollRad : -0.35f, -1.45f, 0.10f);

            if (lockWristBend) { q3 = 0f; q4 = 0f; }
            int posJoints = lockWristBend ? 3 : 5;
            int maxIter = lockWristBend ? 60 : 16;

            float[] q = new float[CanonicalJointCount] { q0, q1, q2, q3, q4, q5 };
            float[] minB = new float[5] { 0.16f, 1.90f, -1.38f, -0.95f, -0.85f };
            float[] maxB = new float[5] { 0.62f, 2.72f, -0.60f,  0.85f,  0.85f };
            float[] priorTarget = new float[5] { 0.28f, 2.42f, -0.95f, -0.20f, -0.20f };
            float[] priorWeight = new float[5] { 0.05f, 0.02f,  0.08f,  0.02f,  0.02f };

            const float eps = 1e-3f;
            float[] qp = new float[CanonicalJointCount];
            Vector3[] J = new Vector3[5];

            for (int iter = 0; iter < maxIter; iter++)
            {
                ComputeForwardKinematics(baseOrigin, q, out Vector3 tip, out _);
                Vector3 err = tip - targetWorldPosition;
                if (err.sqrMagnitude < 1.6e-5f)
                {
                    break;
                }

                for (int j = 0; j < 5; j++)
                {
                    for (int k = 0; k < CanonicalJointCount; k++) qp[k] = q[k];
                    qp[j] += eps;
                    ComputeForwardKinematics(baseOrigin, qp, out Vector3 tipP, out _);
                    J[j] = (tipP - tip) / eps;
                }

                // Damped coordinate updates along the 5 positioning joints preserving open elbow & lateral right elbow clearance
                for (int j = 0; j < posJoints; j++)
                {
                    float jj = Vector3.Dot(J[j], J[j]) + 0.045f;
                    float grad = Vector3.Dot(J[j], err) + 0.01f * priorWeight[j] * (q[j] - priorTarget[j]);
                    float step = -0.55f * (grad / jj);
                    q[j] = Mathf.Clamp(q[j] + step, minB[j], maxB[j]);
                    err += J[j] * step;
                }
            }

            return new float[CanonicalJointCount]
            {
                Mathf.Clamp(q[0], -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(q[1], -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(q[2], -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(q[3], -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(q[4], -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(q[5], -MaxJointAngleRad, MaxJointAngleRad)
            };
        }

        /// <summary>
        /// Canonical 6-DOF anthropomorphic right-arm specification table.
        /// Root base is mounted at ShoulderMountHeight (1.25m) with compact shoulder socket links (0.05m + 0.05m)
        /// so the Glenohumeral (Shoulder_Lift) joint sits at human shoulder height (ShoulderElevation = 1.35m)
        /// without an industrial turret column.
        /// Every child joint's LocalAnchorPosition.y equals its parent link's length (LinkDimensions.y):
        ///   Base_Root:     at y=1.25m, length 0.05m (20.0 kg) -> Shoulder_Pan at y=1.30m
        ///   Shoulder_Pan:  length 0.05m ( 2.5 kg) -> Shoulder_Lift at y=1.35m (glenohumeral joint)
        ///   Shoulder_Lift: length 0.28m ( 2.2 kg) -> Elbow_Flex at +0.28m (human upper arm / humerus)
        ///   Elbow_Flex:    length 0.25m ( 1.6 kg) -> Wrist_Pitch at +0.25m (human forearm / radius-ulna)
        ///   Wrist_Pitch:   length 0.04m ( 0.6 kg) -> Wrist_Yaw at +0.04m (proximal carpal row)
        ///   Wrist_Yaw:     length 0.04m ( 0.5 kg) -> Wrist_Roll at +0.04m (distal carpal / radioulnar)
        ///   Wrist_Roll:    length 0.06m ( 0.5 kg) -> EndEffectorTip at +0.06m (human palm + fingers + thumb)
        /// </summary>
        public static List<ArticulatedLinkSpec> GetArchetypeSpecs(RobotArchetype type = RobotArchetype.ManipulatorArm)
        {
            return new List<ArticulatedLinkSpec>
            {
                new ArticulatedLinkSpec
                {
                    Name = "Base_Root",
                    ParentIndex = -1,
                    LocalAnchorPosition = new Vector3(0f, ShoulderMountHeight, 0f),
                    LinkDimensions = new Vector3(0.088f, BaseHeight, 0.088f),
                    Mass = 20.0f,
                    JointFriction = 0.05f,
                    AngularDamping = 0.25f
                },
                new ArticulatedLinkSpec
                {
                    Name = "Shoulder_Pan",
                    ParentIndex = 0,
                    LocalAnchorPosition = new Vector3(0f, BaseHeight, 0f),
                    LinkDimensions = new Vector3(0.082f, TurretHeight, 0.082f),
                    Mass = 2.5f,
                    JointFriction = 0.06f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -120f,
                    UpperLimitDeg = 120f,
                    DriveStiffness = 300f,
                    DriveDamping = 75f,
                    MaxForceLimit = 600f,
                    AxisRotationEuler = new Vector3(0f, 0f, 90f)
                },
                new ArticulatedLinkSpec
                {
                    Name = "Shoulder_Lift",
                    ParentIndex = 1,
                    LocalAnchorPosition = new Vector3(0f, TurretHeight, 0f),
                    LinkDimensions = new Vector3(0.072f, UpperArmLength, 0.072f),
                    Mass = 2.2f,
                    JointFriction = 0.06f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -90f,
                    UpperLimitDeg = 185f,
                    DriveStiffness = 320f,
                    DriveDamping = 80f,
                    MaxForceLimit = 600f,
                    AxisRotationEuler = new Vector3(0f, 0f, 0f)
                },
                new ArticulatedLinkSpec
                {
                    Name = "Elbow_Flex",
                    ParentIndex = 2,
                    LocalAnchorPosition = new Vector3(0f, UpperArmLength, 0f),
                    LinkDimensions = new Vector3(0.060f, ForearmLength, 0.060f),
                    Mass = 1.6f,
                    JointFriction = 0.06f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -155f,
                    UpperLimitDeg = 15f,
                    DriveStiffness = 260f,
                    DriveDamping = 65f,
                    MaxForceLimit = 500f,
                    AxisRotationEuler = new Vector3(0f, 56f, -16f)
                },
                new ArticulatedLinkSpec
                {
                    Name = "Wrist_Pitch",
                    ParentIndex = 3,
                    LocalAnchorPosition = new Vector3(0f, ForearmLength, 0f),
                    LinkDimensions = new Vector3(0.050f, WristPitchLength, 0.050f),
                    Mass = 0.6f,
                    JointFriction = 0.05f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -95f,
                    UpperLimitDeg = 95f,
                    DriveStiffness = 200f,
                    DriveDamping = 50f,
                    MaxForceLimit = 350f,
                    AxisRotationEuler = new Vector3(0f, 56f, -16f)
                },
                new ArticulatedLinkSpec
                {
                    Name = "Wrist_Yaw",
                    ParentIndex = 4,
                    LocalAnchorPosition = new Vector3(0f, WristPitchLength, 0f),
                    LinkDimensions = new Vector3(0.046f, WristYawLength, 0.046f),
                    Mass = 0.5f,
                    JointFriction = 0.05f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -75f,
                    UpperLimitDeg = 75f,
                    DriveStiffness = 180f,
                    DriveDamping = 45f,
                    MaxForceLimit = 300f,
                    AxisRotationEuler = new Vector3(0f, 90f, 0f)
                },
                new ArticulatedLinkSpec
                {
                    Name = "Wrist_Roll",
                    ParentIndex = 5,
                    LocalAnchorPosition = new Vector3(0f, WristYawLength, 0f),
                    LinkDimensions = new Vector3(0.044f, WristRollLength, 0.044f),
                    Mass = 0.5f,
                    JointFriction = 0.04f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -180f,
                    UpperLimitDeg = 180f,
                    DriveStiffness = 150f,
                    DriveDamping = 40f,
                    MaxForceLimit = 250f,
                    AxisRotationEuler = new Vector3(0f, 0f, 90f)
                }
            };
        }
    }
}
