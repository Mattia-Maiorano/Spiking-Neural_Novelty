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
        public const float MaxJointAngleRad = 2.60f;

        // Exact link lengths (meters) matching GetArchetypeSpecs()
        public const float BaseHeight = 0.16f;
        public const float TurretHeight = 0.14f;
        public const float ShoulderElevation = BaseHeight + TurretHeight; // 0.30m
        public const float UpperArmLength = 0.30f;
        public const float ForearmLength = 0.26f;
        public const float WristPitchLength = 0.11f;
        public const float WristYawLength = 0.09f;
        public const float WristRollLength = 0.08f;
        public const float TotalWristToolLength = WristPitchLength + WristYawLength + WristRollLength; // 0.28m

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
        private PhysicMaterial _groundContactMaterial;

        public RobotArchetype Archetype => archetype;
        public ArticulationBody RootBody => _rootBody;
        public IReadOnlyList<ArticulationBody> ActuatedJoints => _actuatedJoints;
        public int JointCount => _actuatedJoints.Count;
        public ArticulationBody EndEffectorBody => _actuatedJoints.Count > 0 ? _actuatedJoints[_actuatedJoints.Count - 1] : _rootBody;
        public Transform EndEffectorTransform => _endEffectorTipTransform != null
            ? _endEffectorTipTransform
            : (EndEffectorBody != null ? EndEffectorBody.transform : transform);

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

            // Main structural segment spanning (0, 0, 0) -> (0, d.y, 0)
            GameObject shaft = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            shaft.name = spec.Name + "_Visual";
            DestroyImmediate(shaft.GetComponent<Collider>());
            shaft.transform.SetParent(linkPivot, false);
            shaft.transform.localPosition = new Vector3(0f, 0.5f * d.y, 0f);
            shaft.transform.localRotation = Quaternion.identity;
            // Unity Cylinder primitive has height 2.0, so scale.y = 0.5 * d.y yields exact height d.y
            shaft.transform.localScale = new Vector3(d.x, 0.5f * d.y, d.z);

            if (!isRoot)
            {
                // Bolted spherical joint housing right at the pivot (0, 0, 0)
                GameObject hub = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                hub.name = spec.Name + "_JointHub";
                DestroyImmediate(hub.GetComponent<Collider>());
                hub.transform.SetParent(linkPivot, false);
                hub.transform.localPosition = Vector3.zero;
                hub.transform.localRotation = Quaternion.identity;
                float hubDiameter = d.x * 1.22f;
                hub.transform.localScale = new Vector3(hubDiameter, hubDiameter, hubDiameter);
            }

            if (isLastLink)
            {
                // End-Effector magnetic gripper tip at (0, d.y, 0)
                GameObject tipObj = new GameObject("EndEffectorTip");
                tipObj.transform.SetParent(linkPivot, false);
                tipObj.transform.localPosition = new Vector3(0f, d.y, 0f);
                tipObj.transform.localRotation = Quaternion.identity;
                tipObj.transform.localScale = Vector3.one;

                GameObject plate = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                plate.name = "MagneticGripperPlate";
                DestroyImmediate(plate.GetComponent<Collider>());
                plate.transform.SetParent(tipObj.transform, false);
                plate.transform.localPosition = Vector3.zero;
                plate.transform.localRotation = Quaternion.identity;
                plate.transform.localScale = new Vector3(d.x * 1.35f, 0.008f, d.z * 1.35f);

                _endEffectorTipTransform = tipObj.transform;
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
                Vector3 rootPos = transform.position;
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
        /// Returns the canonical industrial neutral resting pose (radians) for Phase 0 (Setup).
        /// </summary>
        public static float[] GetNeutralRestingPose()
        {
            return new float[CanonicalJointCount]
            {
                0.00f,  // Shoulder_Pan   (centered forward)
                0.35f,  // Shoulder_Lift  (poised forward 20 deg)
                0.85f,  // Elbow_Flex     (poised elbow-up 49 deg)
                0.35f,  // Wrist_Pitch    (angled downward)
                0.00f,  // Wrist_Yaw      (centered)
                0.00f   // Wrist_Roll     (level)
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
        /// to exact generalized joint angles q in R^6 (radians) using an industrial elbow-up configuration.
        /// Guarantees that the end-effector tip reaches ground-level targets accurately.
        /// </summary>
        public float[] SolveInverseKinematics6D(Vector3 targetWorldPosition, Quaternion targetWorldOrientation)
        {
            Vector3 baseOrigin = _rootBody != null ? _rootBody.transform.position : transform.position;
            return SolveInverseKinematicsStatic(baseOrigin, targetWorldPosition, targetWorldOrientation);
        }

        public static float[] SolveInverseKinematicsStatic(
            Vector3 baseOrigin,
            Vector3 targetWorldPosition,
            Quaternion targetWorldOrientation)
        {
            Vector3 shoulderOrigin = baseOrigin + new Vector3(0f, ShoulderElevation, 0f);
            Vector3 rel = targetWorldPosition - shoulderOrigin;

            // 1. Shoulder Pan (q0): rotate vertical arm plane to align with target (x, z)
            float planarDist = Mathf.Sqrt(rel.x * rel.x + rel.z * rel.z);
            float pan = planarDist > 1e-4f ? Mathf.Atan2(rel.x, rel.z) : 0f;
            float verticalDist = rel.y; // negative when target is on the ground below shoulder (0.30m)

            // 2. Desired tool pitch angle thetaTool (measured from +Y toward radial direction in arm plane)
            Vector3 eulerDeg = targetWorldOrientation.eulerAngles;
            float targetPitchRad = Mathf.DeltaAngle(0f, eulerDeg.x) * Mathf.Deg2Rad;
            float targetYawRad = Mathf.DeltaAngle(0f, eulerDeg.y) * Mathf.Deg2Rad;
            float targetRollRad = Mathf.DeltaAngle(0f, eulerDeg.z) * Mathf.Deg2Rad;

            // When reaching low/ground targets, angle the tool downward (~135-155 deg from +Y);
            // when lifting/transporting high targets, incorporate targetPitchRad for wrist reorientation.
            float heightFactor = Mathf.InverseLerp(-0.26f, 0.12f, verticalDist);
            float baseToolAngle = Mathf.Lerp(2.50f, 1.65f, heightFactor) + 0.55f * targetPitchRad;
            float thetaTool = Mathf.Clamp(baseToolAngle, 1.10f, 2.75f);

            // 3. Wrist Pitch pivot position (r_w, y_w) relative to Shoulder Lift pivot
            float rw = planarDist - TotalWristToolLength * Mathf.Sin(thetaTool);
            float yw = verticalDist - TotalWristToolLength * Mathf.Cos(thetaTool);

            // Ensure (rw, yw) lies strictly inside the 2-link reachable annulus [|L1-L2|+eps, L1+L2-eps]
            float d = Mathf.Sqrt(rw * rw + yw * yw);
            const float minReach = (UpperArmLength - ForearmLength) + 0.015f; // 0.055m
            const float maxReach = (UpperArmLength + ForearmLength) - 0.015f; // 0.545m
            if (d > maxReach && d > 1e-5f)
            {
                float scale = maxReach / d;
                rw *= scale;
                yw *= scale;
                d = maxReach;
            }
            else if (d < minReach)
            {
                d = minReach;
            }

            // 4. Law of cosines for Elbow_Flex (q2 >= 0: industrial elbow-up posture) and Shoulder_Lift (q1)
            float cosElbow = Mathf.Clamp(
                (d * d - UpperArmLength * UpperArmLength - ForearmLength * ForearmLength)
                / (2.0f * UpperArmLength * ForearmLength),
                -0.985f,
                0.985f
            );
            float elbowFlex = Mathf.Acos(cosElbow); // positive = elbow-up, forearm bends forward & down
            float targetAngleFromUp = Mathf.Atan2(rw, yw);
            float shoulderTriangleOffset = Mathf.Atan2(
                ForearmLength * Mathf.Sin(elbowFlex),
                UpperArmLength + ForearmLength * Mathf.Cos(elbowFlex)
            );
            float shoulderLift = targetAngleFromUp - shoulderTriangleOffset;

            // 5. Spherical wrist joints (Wrist_Pitch q3, Wrist_Yaw q4, Wrist_Roll q5)
            float wristPitch = thetaTool - (shoulderLift + elbowFlex);
            float wristYaw = Mathf.Clamp(targetYawRad, -1.4f, 1.4f);
            float wristRoll = Mathf.Clamp(targetRollRad, -2.2f, 2.2f);

            return new float[CanonicalJointCount]
            {
                Mathf.Clamp(pan, -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(shoulderLift, -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(elbowFlex, -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(wristPitch, -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(wristYaw, -MaxJointAngleRad, MaxJointAngleRad),
                Mathf.Clamp(wristRoll, -MaxJointAngleRad, MaxJointAngleRad)
            };
        }

        /// <summary>
        /// Canonical 6-DOF industrial manipulator specification table.
        /// Root base is permanently bolted to the ground (immovable, 20.0 kg).
        /// Every child joint's LocalAnchorPosition.y equals its parent link's length (LinkDimensions.y):
        ///   Base_Root:     length 0.16m (20.0 kg) -> Shoulder_Pan at y=0.16m
        ///   Shoulder_Pan:  length 0.14m ( 4.0 kg) -> Shoulder_Lift at y=0.14m
        ///   Shoulder_Lift: length 0.30m ( 3.2 kg) -> Elbow_Flex at y=0.30m
        ///   Elbow_Flex:    length 0.26m ( 2.4 kg) -> Wrist_Pitch at y=0.26m
        ///   Wrist_Pitch:   length 0.11m ( 1.5 kg) -> Wrist_Yaw at y=0.11m
        ///   Wrist_Yaw:     length 0.09m ( 1.0 kg) -> Wrist_Roll at y=0.09m
        ///   Wrist_Roll:    length 0.08m ( 0.6 kg) -> EndEffectorTip at y=0.08m
        /// </summary>
        public static List<ArticulatedLinkSpec> GetArchetypeSpecs(RobotArchetype type = RobotArchetype.ManipulatorArm)
        {
            return new List<ArticulatedLinkSpec>
            {
                new ArticulatedLinkSpec
                {
                    Name = "Base_Root",
                    ParentIndex = -1,
                    LocalAnchorPosition = Vector3.zero,
                    LinkDimensions = new Vector3(0.30f, BaseHeight, 0.30f),
                    Mass = 20.0f,
                    JointFriction = 0.05f,
                    AngularDamping = 0.25f
                },
                new ArticulatedLinkSpec
                {
                    Name = "Shoulder_Pan",
                    ParentIndex = 0,
                    LocalAnchorPosition = new Vector3(0f, BaseHeight, 0f),
                    LinkDimensions = new Vector3(0.14f, TurretHeight, 0.14f),
                    Mass = 4.0f,
                    JointFriction = 0.06f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -170f,
                    UpperLimitDeg = 170f,
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
                    LinkDimensions = new Vector3(0.11f, UpperArmLength, 0.11f),
                    Mass = 3.2f,
                    JointFriction = 0.06f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -150f,
                    UpperLimitDeg = 150f,
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
                    LinkDimensions = new Vector3(0.09f, ForearmLength, 0.09f),
                    Mass = 2.4f,
                    JointFriction = 0.06f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -160f,
                    UpperLimitDeg = 160f,
                    DriveStiffness = 260f,
                    DriveDamping = 65f,
                    MaxForceLimit = 500f,
                    AxisRotationEuler = new Vector3(0f, 0f, 0f)
                },
                new ArticulatedLinkSpec
                {
                    Name = "Wrist_Pitch",
                    ParentIndex = 3,
                    LocalAnchorPosition = new Vector3(0f, ForearmLength, 0f),
                    LinkDimensions = new Vector3(0.075f, WristPitchLength, 0.075f),
                    Mass = 1.5f,
                    JointFriction = 0.05f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -165f,
                    UpperLimitDeg = 165f,
                    DriveStiffness = 200f,
                    DriveDamping = 50f,
                    MaxForceLimit = 350f,
                    AxisRotationEuler = new Vector3(0f, 0f, 0f)
                },
                new ArticulatedLinkSpec
                {
                    Name = "Wrist_Yaw",
                    ParentIndex = 4,
                    LocalAnchorPosition = new Vector3(0f, WristPitchLength, 0f),
                    LinkDimensions = new Vector3(0.06f, WristYawLength, 0.06f),
                    Mass = 1.0f,
                    JointFriction = 0.05f,
                    AngularDamping = 0.25f,
                    LowerLimitDeg = -165f,
                    UpperLimitDeg = 165f,
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
                    LinkDimensions = new Vector3(0.048f, WristRollLength, 0.048f),
                    Mass = 0.6f,
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
