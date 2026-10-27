using System;
using UnityEngine;

namespace SPWM.Robotics
{
    /// <summary>
    /// Primitive payload shapes for inertial and visual variance during active perception grasping.
    /// </summary>
    public enum PayloadPrimitiveShape
    {
        Sphere = 0,
        ElongatedCylinder = 1,
        Box = 2
    }

    /// <summary>
    /// Episode specification for the procedurally spawned payload and its 6D target pose.
    /// </summary>
    [Serializable]
    public struct SpawnedPayloadSpec
    {
        public PayloadPrimitiveShape Shape;
        public float Mass;
        public Vector3 Scale;
        public Vector3 StartPosition;
        public Quaternion StartRotation;
        public Vector3 TargetPosition;
        public Quaternion TargetRotation;

        /// <summary>
        /// Vertical half-height of the payload as it rests flat on the ground at spawn.
        /// </summary>
        public float HalfHeight
        {
            get
            {
                if (Shape == PayloadPrimitiveShape.ElongatedCylinder)
                {
                    // When lying flat on the ground (rotated 90 deg about Z), vertical diameter is Scale.x
                    return 0.5f * Scale.x;
                }
                return 0.5f * Scale.y;
            }
        }

        /// <summary>
        /// World-space 3D coordinates of the top grasp surface of the object sitting on the ground.
        /// </summary>
        public Vector3 GraspPoint => StartPosition + new Vector3(0f, HalfHeight, 0f);
    }

    /// <summary>
    /// Collision-triggered or controller-triggered rigid magnetic grasp lock component.
    /// Instantiates an unbreakable FixedJoint constraint upon end-effector contact,
    /// physically locking the payload directly to the manipulator's end-effector tip so
    /// the payload's mass and rotational inertia tensor pull on the arm's joints.
    /// </summary>
    public class PayloadGraspConstraint : MonoBehaviour
    {
        private ArticulationBody _targetEndEffector;
        private Transform _targetTipTransform;
        private FixedJoint _fixedJoint;
        private bool _isAttached;

        public bool IsAttached => _isAttached;
        public FixedJoint ActiveJoint => _fixedJoint;

        public void ConfigureTargetEndEffector(ArticulationBody endEffector, Transform tipTransform = null)
        {
            _targetEndEffector = endEffector;
            _targetTipTransform = tipTransform;
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (_isAttached || _targetEndEffector == null)
            {
                return;
            }

            ArticulationBody hitAb = collision.collider.GetComponentInParent<ArticulationBody>();
            if (hitAb != null && hitAb == _targetEndEffector)
            {
                AttachRigidly(_targetEndEffector, _targetTipTransform);
            }
        }

        /// <summary>
        /// Instantiates an instant rigid FixedJoint lock (like an industrial electromagnet)
        /// coupling the payload Rigidbody directly to the manipulator's end-effector ArticulationBody.
        /// </summary>
        public FixedJoint AttachRigidly(ArticulationBody endEffector, Transform tipTransform = null)
        {
            if (_isAttached && _fixedJoint != null)
            {
                return _fixedJoint;
            }

            _targetEndEffector = endEffector;
            if (tipTransform != null)
            {
                _targetTipTransform = tipTransform;
            }

            // Snap payload flush beneath the magnetic gripper plate at the end-effector tip
            if (_targetTipTransform != null)
            {
                float halfExtent = 0.5f * Mathf.Min(transform.localScale.x, transform.localScale.y);
                transform.position = _targetTipTransform.position - _targetTipTransform.up * halfExtent;
            }

            Rigidbody rb = GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.useGravity = true;
                rb.isKinematic = false;
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.solverIterations = 24;
                rb.solverVelocityIterations = 8;
            }

            _fixedJoint = gameObject.AddComponent<FixedJoint>();
            if (endEffector != null)
            {
                _fixedJoint.connectedArticulationBody = endEffector;
            }
            _fixedJoint.breakForce = float.PositiveInfinity;
            _fixedJoint.breakTorque = float.PositiveInfinity;
            _fixedJoint.enableCollision = false;
            _fixedJoint.enablePreprocessing = false;
            _isAttached = true;
            return _fixedJoint;
        }
    }

    /// <summary>
    /// Procedural Object Spawner for SPWM-v0.5.0 6-DOF Active Perception Grasping.
    /// Generates a heavy target payload resting on the physical ground plane at the start of each episode:
    ///   - Randomized starting (x, z) coordinates on the ground within the reachable workspace
    ///   - Randomized final 6D target pose (3D translation + 3D upright twisted orientation)
    ///   - Randomized geometry across primitive shapes (spheres, elongated cylinders, boxes)
    ///   - Randomized heavy mass (1.2 kg to 4.5 kg) and full rotational inertia tensors
    ///   - Instant rigid magnetic FixedJoint lock upon end-effector contact
    /// </summary>
    public class ProceduralObjectSpawner : MonoBehaviour
    {
        [Header("Workspace Bounds (Meters from Base)")]
        [Tooltip("Minimum horizontal distance from robot base to initial ground object spawn.")]
        [SerializeField] private float minReachRadius = 0.22f;
        [Tooltip("Maximum horizontal distance from robot base to initial ground object spawn.")]
        [SerializeField] private float maxReachRadius = 0.52f;
        [Tooltip("Minimum azimuth angle (radians) for initial ground object spawn.")]
        [SerializeField] private float minStartAzimuth = -0.75f;
        [Tooltip("Maximum azimuth angle (radians) for initial ground object spawn.")]
        [SerializeField] private float maxStartAzimuth = 0.75f;
        [SerializeField] private float minTargetHeight = 0.24f;
        [SerializeField] private float maxTargetHeight = 0.38f;

        [Header("Inertial & Visual Variance")]
        [SerializeField] private float minPayloadMassKg = 1.20f;
        [SerializeField] private float maxPayloadMassKg = 4.50f;

        private GameObject _groundPlaneObject;
        private GameObject _activePayloadObject;
        private Rigidbody _activePayloadRigidbody;
        private PayloadGraspConstraint _activeGraspConstraint;
        private SpawnedPayloadSpec _currentSpec;

        public GameObject ActivePayloadObject => _activePayloadObject;
        public Rigidbody ActivePayloadRigidbody => _activePayloadRigidbody;
        public PayloadGraspConstraint ActiveGraspConstraint => _activeGraspConstraint;
        public SpawnedPayloadSpec CurrentSpec => _currentSpec;
        public bool IsPayloadGrasped => _activeGraspConstraint != null && _activeGraspConstraint.IsAttached;

        /// <summary>
        /// Deterministically computes the episode payload specification for a given seed and base origin.
        /// Used by both physical simulation and offline trajectory visualization.
        /// Wildly varies the initial distance (radius) and approach azimuth to break predictor temporal timing memorization.
        /// </summary>
        public static SpawnedPayloadSpec GenerateDeterministicSpec(
            int episodeSeed,
            Vector3 baseOrigin,
            float minRadius = 0.22f,
            float maxRadius = 0.52f,
            float minHeight = 0.24f,
            float maxHeight = 0.38f,
            float minMass = 1.20f,
            float maxMass = 4.50f,
            float minStartAzimuth = -0.75f,
            float maxStartAzimuth = 0.75f)
        {
            System.Random rng = new System.Random(episodeSeed * 7919 + 104729);

            var shapes = (PayloadPrimitiveShape[])Enum.GetValues(typeof(PayloadPrimitiveShape));
            PayloadPrimitiveShape shape = shapes[rng.Next(shapes.Length)];
            float mass = Mathf.Lerp(minMass, maxMass, (float)rng.NextDouble());

            Vector3 scale;
            float halfHeightOnGround;
            Quaternion flatStartRotation;
            float startYawDeg = Mathf.Lerp(-35f, 35f, (float)rng.NextDouble());

            switch (shape)
            {
                case PayloadPrimitiveShape.Sphere:
                    float diameter = Mathf.Lerp(0.08f, 0.12f, (float)rng.NextDouble());
                    scale = new Vector3(diameter, diameter, diameter);
                    halfHeightOnGround = 0.5f * diameter;
                    flatStartRotation = Quaternion.Euler(0f, startYawDeg, 0f);
                    break;

                case PayloadPrimitiveShape.ElongatedCylinder:
                    // Lay cylinder flat on the ground (rotated 90 deg around Z) so it can be lifted upright during transport
                    float cylDiameter = Mathf.Lerp(0.06f, 0.09f, (float)rng.NextDouble());
                    float cylHalfLength = Mathf.Lerp(0.09f, 0.14f, (float)rng.NextDouble());
                    scale = new Vector3(cylDiameter, cylHalfLength, cylDiameter);
                    halfHeightOnGround = 0.5f * cylDiameter;
                    flatStartRotation = Quaternion.Euler(0f, startYawDeg, 90f);
                    break;

                default:
                    // Flat heavy industrial block resting on the ground
                    float width = Mathf.Lerp(0.10f, 0.15f, (float)rng.NextDouble());
                    float thickness = Mathf.Lerp(0.06f, 0.08f, (float)rng.NextDouble());
                    float depth = Mathf.Lerp(0.09f, 0.14f, (float)rng.NextDouble());
                    scale = new Vector3(width, thickness, depth);
                    halfHeightOnGround = 0.5f * thickness;
                    flatStartRotation = Quaternion.Euler(0f, startYawDeg, 0f);
                    break;
            }

            // 2. Starting (x, y, z) sitting directly on the ground plane (y = baseOrigin.y + halfHeightOnGround)
            // Wildly varied distance and azimuth from the base
            float startRadius = Mathf.Lerp(minRadius, maxRadius, (float)rng.NextDouble());
            float startAzimuth = Mathf.Lerp(minStartAzimuth, maxStartAzimuth, (float)rng.NextDouble());
            if (Mathf.Abs(startAzimuth) < 0.12f)
            {
                startAzimuth = startAzimuth >= 0f ? 0.18f : -0.18f;
            }

            Vector3 startPos = baseOrigin + new Vector3(
                startRadius * Mathf.Sin(startAzimuth),
                halfHeightOnGround,
                startRadius * Mathf.Cos(startAzimuth)
            );

            // 3. Final 6D target pose at an elevated workspace coordinate on the opposite azimuth,
            //    with a deliberate 80-90 deg upright wrist reorientation (picking up flat -> lifting upright).
            float targetRadius = Mathf.Lerp(minRadius, maxRadius - 0.02f, (float)rng.NextDouble());
            float targetAzimuth = -Mathf.Sign(startAzimuth) * Mathf.Lerp(0.32f, 0.65f, (float)rng.NextDouble());
            float targetHeight = Mathf.Lerp(minHeight, maxHeight, (float)rng.NextDouble());
            Vector3 targetPos = baseOrigin + new Vector3(
                targetRadius * Mathf.Sin(targetAzimuth),
                targetHeight,
                targetRadius * Mathf.Cos(targetAzimuth)
            );

            Quaternion targetRot = Quaternion.Euler(
                Mathf.Lerp(55f, 85f, (float)rng.NextDouble()),
                startYawDeg + Mathf.Sign(targetAzimuth) * Mathf.Lerp(40f, 75f, (float)rng.NextDouble()),
                Mathf.Lerp(-85f, -55f, (float)rng.NextDouble())
            );

            return new SpawnedPayloadSpec
            {
                Shape = shape,
                Mass = mass,
                Scale = scale,
                StartPosition = startPos,
                StartRotation = flatStartRotation,
                TargetPosition = targetPos,
                TargetRotation = targetRot
            };
        }

        private void EnsureGroundPlane(Vector3 baseOrigin, ArticulatedRobotRig rig)
        {
            if (_groundPlaneObject == null)
            {
                _groundPlaneObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
                _groundPlaneObject.name = "IndustrialGroundFloor";
                _groundPlaneObject.transform.SetParent(transform, true);
                _groundPlaneObject.transform.localScale = new Vector3(4.0f, 0.04f, 4.0f);
                var renderer = _groundPlaneObject.GetComponent<Renderer>();
                if (renderer != null)
                {
                    Shader s = Shader.Find("Standard");
                    if (s != null)
                    {
                        var mat = new Material(s)
                        {
                            name = "GroundFloorMat (runtime)",
                            color = new Color(0.14f, 0.16f, 0.19f)
                        };
                        mat.SetFloat("_Glossiness", 0.15f);
                        renderer.sharedMaterial = mat;
                    }
                }
            }

            _groundPlaneObject.transform.position = new Vector3(baseOrigin.x, baseOrigin.y - 0.02f, baseOrigin.z);
            _groundPlaneObject.transform.rotation = Quaternion.identity;

            // Ignore collision between the ground floor and the bolted robot arm links so the gripper tip
            // can cleanly reach ground-level objects without fighting floor contact forces.
            Collider floorCollider = _groundPlaneObject.GetComponent<Collider>();
            if (floorCollider != null && rig != null)
            {
                Collider[] armColliders = rig.GetComponentsInChildren<Collider>();
                for (int i = 0; i < armColliders.Length; i++)
                {
                    if (armColliders[i] != null && armColliders[i] != floorCollider)
                    {
                        Physics.IgnoreCollision(floorCollider, armColliders[i], true);
                    }
                }
            }
        }

        /// <summary>
        /// Spawns a new randomized target payload resting on the ground for the episode seeded by episodeSeed.
        /// </summary>
        public SpawnedPayloadSpec SpawnEpisodePayload(int episodeSeed, ArticulatedRobotRig rig = null)
        {
            ClearActivePayload();

            Vector3 baseOrigin = rig != null ? rig.transform.position : transform.position;
            EnsureGroundPlane(baseOrigin, rig);

            _currentSpec = GenerateDeterministicSpec(
                episodeSeed,
                baseOrigin,
                minReachRadius,
                maxReachRadius,
                minTargetHeight,
                maxTargetHeight,
                minPayloadMassKg,
                maxPayloadMassKg,
                minStartAzimuth,
                maxStartAzimuth
            );

            PrimitiveType unityPrimitive;
            switch (_currentSpec.Shape)
            {
                case PayloadPrimitiveShape.Sphere:
                    unityPrimitive = PrimitiveType.Sphere;
                    break;
                case PayloadPrimitiveShape.ElongatedCylinder:
                    unityPrimitive = PrimitiveType.Cylinder;
                    break;
                default:
                    unityPrimitive = PrimitiveType.Cube;
                    break;
            }

            _activePayloadObject = GameObject.CreatePrimitive(unityPrimitive);
            _activePayloadObject.name = $"TargetPayload_{_currentSpec.Shape}_{episodeSeed:D4}";
            _activePayloadObject.transform.SetParent(transform, true);
            _activePayloadObject.transform.position = _currentSpec.StartPosition;
            _activePayloadObject.transform.rotation = _currentSpec.StartRotation;
            _activePayloadObject.transform.localScale = _currentSpec.Scale;

            var payloadRenderer = _activePayloadObject.GetComponent<Renderer>();
            if (payloadRenderer != null)
            {
                Shader s = Shader.Find("Standard");
                if (s != null)
                {
                    var mat = new Material(s)
                    {
                        name = "PayloadMat (runtime)",
                        color = new Color(0.88f, 0.42f, 0.14f)
                    };
                    mat.SetFloat("_Glossiness", 0.35f);
                    mat.SetFloat("_Metallic", 0.25f);
                    payloadRenderer.sharedMaterial = mat;
                }
            }

            _activePayloadRigidbody = _activePayloadObject.AddComponent<Rigidbody>();
            _activePayloadRigidbody.mass = _currentSpec.Mass;
            _activePayloadRigidbody.useGravity = true;
            _activePayloadRigidbody.linearDamping = 0.05f;
            _activePayloadRigidbody.angularDamping = 0.05f;
            _activePayloadRigidbody.ResetInertiaTensor();

            _activeGraspConstraint = _activePayloadObject.AddComponent<PayloadGraspConstraint>();
            if (rig != null && rig.EndEffectorBody != null)
            {
                _activeGraspConstraint.ConfigureTargetEndEffector(rig.EndEffectorBody, rig.EndEffectorTransform);
            }

            return _currentSpec;
        }

        /// <summary>
        /// Ensures an instant rigid FixedJoint lock between the active payload and the manipulator end-effector.
        /// </summary>
        public FixedJoint EnsureRigidGrasp(ArticulatedRobotRig rig)
        {
            if (_activeGraspConstraint == null || rig == null)
            {
                return null;
            }
            return _activeGraspConstraint.AttachRigidly(rig.EndEffectorBody, rig.EndEffectorTransform);
        }

        public void ClearActivePayload()
        {
            if (_activePayloadObject != null)
            {
                DestroyImmediate(_activePayloadObject);
                _activePayloadObject = null;
                _activePayloadRigidbody = null;
                _activeGraspConstraint = null;
            }
        }
    }
}
