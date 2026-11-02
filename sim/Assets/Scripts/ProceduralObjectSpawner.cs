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
    /// Three distinct vertical elevation planes for anthropomorphic target spawning directly in front of the head:
    ///   - Low:  Lower tabletop workspace plane below shoulder level (~1.04m above ground, -0.31m relative to shoulder)
    ///   - Mid:  Mid manipulation workspace plane below shoulder level (~1.14m above ground, -0.21m relative to shoulder)
    ///   - High: Upper chest/reach workspace plane (~1.26m above ground, -0.09m relative to shoulder)
    /// </summary>
    public enum TargetElevationPlane
    {
        Low = 0,
        Mid = 1,
        High = 2
    }

    /// <summary>
    /// Episode specification for the procedurally spawned payload and its 6D target pose.
    /// </summary>
    [Serializable]
    public struct SpawnedPayloadSpec
    {
        public PayloadPrimitiveShape Shape;
        public TargetElevationPlane ElevationPlane;
        public float Mass;
        public Vector3 Scale;
        public Vector3 StartPosition;
        public Quaternion StartRotation;
        public Vector3 TargetPosition;
        public Quaternion TargetRotation;

        /// <summary>
        /// Vertical half-height of the payload.
        /// </summary>
        public float HalfHeight
        {
            get
            {
                if (Shape == PayloadPrimitiveShape.ElongatedCylinder)
                {
                    return 0.5f * Scale.x;
                }
                return 0.5f * Scale.y;
            }
        }

        /// <summary>
        /// Half-depth of the payload along the forward (+Z) approach axis.
        /// </summary>
        public float HalfDepth
        {
            get
            {
                if (Shape == PayloadPrimitiveShape.Sphere || Shape == PayloadPrimitiveShape.ElongatedCylinder)
                {
                    return 0.5f * Scale.x;
                }
                return 0.5f * Scale.z;
            }
        }

        /// <summary>
        /// World-space 3D coordinates of the proximal grasp surface of the target object facing the approaching hand.
        /// </summary>
        public Vector3 GraspPoint => StartPosition - new Vector3(0f, 0f, HalfDepth);
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

            // Snap payload into the humanoid palm & curved fingers at the end-effector tip
            if (_targetTipTransform != null)
            {
                float halfExtent = 0.5f * Mathf.Min(transform.localScale.x, transform.localScale.z);
                transform.position = _targetTipTransform.position + _targetTipTransform.rotation * new Vector3(0f, 0.035f + 0.65f * halfExtent, 0.024f);
            }

            Rigidbody rb = GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.constraints = RigidbodyConstraints.None;
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
    /// Procedural Object Spawner for Anthropomorphic 6-DOF Active Perception Grasping.
    /// Places the target payload directly in front of the human torso/head along the forward (+Z) axis,
    /// with logic to randomize or toggle between three distinct vertical elevation planes:
    ///   - Low plane  (~1.04m, tabletop level below shoulder)
    ///   - Mid plane  (~1.14m, natural human manipulation level)
    ///   - High plane (~1.26m, upper chest/reach level)
    /// Preserves randomized geometry (spheres, elongated cylinders, boxes), heavy mass (1.2 kg to 4.5 kg),
    /// rotational inertia tensors, and instant rigid FixedJoint lock upon end-effector contact.
    /// </summary>
    public class ProceduralObjectSpawner : MonoBehaviour
    {
        public const float DefaultLowPlaneHeight = 1.04f;   // ShoulderElevation (1.35m) - 0.31m (tabletop level)
        public const float DefaultMidPlaneHeight = 1.14f;   // ShoulderElevation (1.35m) - 0.21m (natural human manipulation level)
        public const float DefaultHighPlaneHeight = 1.26f;  // ShoulderElevation (1.35m) - 0.09m (chest/upper reach level)

        [Header("Anthropomorphic Forward Workspace Bounds")]
        [Tooltip("Minimum forward distance (+Z) from shoulder/head plane to target spawn.")]
        [SerializeField] private float minReachRadius = 0.42f;
        [Tooltip("Maximum forward distance (+Z) from shoulder/head plane to target spawn.")]
        [SerializeField] private float maxReachRadius = 0.52f;
        [Tooltip("Maximum lateral jitter (meters) around the head's forward axis so target stays directly in front of the head.")]
        [SerializeField] private float maxLateralJitter = 0.04f;
        [Tooltip("Minimum azimuth angle (radians) for transport target offset.")]
        [SerializeField] private float minStartAzimuth = -0.10f;
        [Tooltip("Maximum azimuth angle (radians) for transport target offset.")]
        [SerializeField] private float maxStartAzimuth = 0.10f;

        [Header("Three Vertical Elevation Planes (Low / Mid / High)")]
        [Tooltip("If true, deterministically randomizes between Low, Mid, and High vertical planes each episode. If false, uses activeElevationPlane.")]
        [SerializeField] private bool randomizeElevationPlane = true;
        [Tooltip("Active vertical elevation plane when randomizeElevationPlane is false (or toggled via ToggleElevationPlane).")]
        [SerializeField] private TargetElevationPlane activeElevationPlane = TargetElevationPlane.Mid;
        [Tooltip("Vertical elevation (meters above ground) for the Low plane.")]
        [SerializeField] private float lowPlaneElevation = DefaultLowPlaneHeight;
        [Tooltip("Vertical elevation (meters above ground) for the Mid (shoulder-level) plane.")]
        [SerializeField] private float midPlaneElevation = DefaultMidPlaneHeight;
        [Tooltip("Vertical elevation (meters above ground) for the High (eye-level) plane.")]
        [SerializeField] private float highPlaneElevation = DefaultHighPlaneHeight;
        [Tooltip("Small within-plane vertical variance (meters) to prevent exact coordinate memorization while keeping planes distinct.")]
        [SerializeField] private float planeVerticalJitter = 0.015f;

        [Header("Transport Target Bounds")]
        [SerializeField] private float minTargetHeight = DefaultLowPlaneHeight;
        [SerializeField] private float maxTargetHeight = DefaultHighPlaneHeight;

        [Header("Inertial & Visual Variance")]
        [SerializeField] private float minPayloadMassKg = 1.20f;
        [SerializeField] private float maxPayloadMassKg = 4.50f;

        private GameObject _groundPlaneObject;
        private GameObject _activePayloadObject;
        private Rigidbody _activePayloadRigidbody;
        private PayloadGraspConstraint _activeGraspConstraint;
        private SpawnedPayloadSpec _currentSpec;
        private int _lastEpisodeSeed = 42;
        private ArticulatedRobotRig _lastRig;

        public GameObject ActivePayloadObject => _activePayloadObject;
        public Rigidbody ActivePayloadRigidbody => _activePayloadRigidbody;
        public PayloadGraspConstraint ActiveGraspConstraint => _activeGraspConstraint;
        public SpawnedPayloadSpec CurrentSpec => _currentSpec;
        public bool IsPayloadGrasped => _activeGraspConstraint != null && _activeGraspConstraint.IsAttached;
        public bool RandomizeElevationPlane
        {
            get => randomizeElevationPlane;
            set => randomizeElevationPlane = value;
        }
        public TargetElevationPlane ActiveElevationPlane
        {
            get => activeElevationPlane;
            set => activeElevationPlane = value;
        }

        /// <summary>
        /// Returns the vertical height (meters above baseOrigin.y) for the given elevation plane (Low, Mid, High).
        /// </summary>
        public static float GetPlaneElevationMeters(
            TargetElevationPlane plane,
            float lowHeight = DefaultLowPlaneHeight,
            float midHeight = DefaultMidPlaneHeight,
            float highHeight = DefaultHighPlaneHeight)
        {
            switch (plane)
            {
                case TargetElevationPlane.Low:
                    return lowHeight;
                case TargetElevationPlane.High:
                    return highHeight;
                case TargetElevationPlane.Mid:
                default:
                    return midHeight;
            }
        }

        /// <summary>
        /// Toggles sequentially between Low -> Mid -> High vertical target elevation planes
        /// and updates the active target object in front of the head if one is currently spawned.
        /// </summary>
        [ContextMenu("Toggle Target Elevation Plane (Low / Mid / High)")]
        public TargetElevationPlane ToggleElevationPlane()
        {
            randomizeElevationPlane = false;
            activeElevationPlane = (TargetElevationPlane)(((int)activeElevationPlane + 1) % 3);
            if (_activePayloadObject != null)
            {
                SpawnEpisodePayload(_lastEpisodeSeed, _lastRig);
            }
            return activeElevationPlane;
        }

        /// <summary>
        /// Explicitly sets the target elevation plane (Low, Mid, or High) and optionally disables plane randomization.
        /// </summary>
        public void SetElevationPlane(TargetElevationPlane plane, bool disableRandomization = true)
        {
            activeElevationPlane = plane;
            if (disableRandomization)
            {
                randomizeElevationPlane = false;
            }
            if (_activePayloadObject != null)
            {
                SpawnEpisodePayload(_lastEpisodeSeed, _lastRig);
            }
        }

        /// <summary>
        /// Deterministically computes the episode payload specification for a given seed and base origin.
        /// Places the target object directly in front of the head along the forward (+Z) axis at one of
        /// three distinct vertical elevations (Low, Mid, or High plane), while varying forward reach distance,
        /// shape, and mass to prevent predictor temporal/spatial memorization.
        /// </summary>
        public static SpawnedPayloadSpec GenerateDeterministicSpec(
            int episodeSeed,
            Vector3 baseOrigin,
            float minRadius = 0.42f,
            float maxRadius = 0.52f,
            float minHeight = DefaultLowPlaneHeight,
            float maxHeight = DefaultHighPlaneHeight,
            float minMass = 1.20f,
            float maxMass = 4.50f,
            float minStartAzimuth = -0.10f,
            float maxStartAzimuth = 0.10f,
            TargetElevationPlane? forcedElevationPlane = null,
            float lowPlaneHeight = DefaultLowPlaneHeight,
            float midPlaneHeight = DefaultMidPlaneHeight,
            float highPlaneHeight = DefaultHighPlaneHeight,
            float lateralJitterMax = 0.04f,
            float verticalJitterMax = 0.015f)
        {
            System.Random rng = new System.Random(episodeSeed * 7919 + 104729);

            var shapes = (PayloadPrimitiveShape[])Enum.GetValues(typeof(PayloadPrimitiveShape));
            PayloadPrimitiveShape shape = shapes[rng.Next(shapes.Length)];
            float mass = Mathf.Lerp(minMass, maxMass, (float)rng.NextDouble());

            // 1. Select vertical elevation plane: Low, Mid, or High
            TargetElevationPlane elevationPlane = forcedElevationPlane.HasValue
                ? forcedElevationPlane.Value
                : (TargetElevationPlane)rng.Next(3);

            Vector3 scale;
            Quaternion startRotation;
            float startYawDeg = Mathf.Lerp(-20f, 20f, (float)rng.NextDouble());

            switch (shape)
            {
                case PayloadPrimitiveShape.Sphere:
                    float diameter = Mathf.Lerp(0.08f, 0.11f, (float)rng.NextDouble());
                    scale = new Vector3(diameter, diameter, diameter);
                    startRotation = Quaternion.Euler(0f, startYawDeg, 0f);
                    break;

                case PayloadPrimitiveShape.ElongatedCylinder:
                    float cylDiameter = Mathf.Lerp(0.06f, 0.085f, (float)rng.NextDouble());
                    float cylHalfLength = Mathf.Lerp(0.08f, 0.12f, (float)rng.NextDouble());
                    scale = new Vector3(cylDiameter, cylHalfLength, cylDiameter);
                    startRotation = Quaternion.Euler(0f, startYawDeg, 90f);
                    break;

                default:
                    float width = Mathf.Lerp(0.09f, 0.13f, (float)rng.NextDouble());
                    float thickness = Mathf.Lerp(0.06f, 0.085f, (float)rng.NextDouble());
                    float depth = Mathf.Lerp(0.08f, 0.12f, (float)rng.NextDouble());
                    scale = new Vector3(width, thickness, depth);
                    startRotation = Quaternion.Euler(0f, startYawDeg, 0f);
                    break;
            }

            // 2. Starting (x, y, z) placed strictly on the left in front of the head
            //    (x ≈ -0.17m..-0.20m) at the chosen vertical elevation plane (Low, Mid, or High).
            //    Reaching to the left enforces the human-like reaching behavior: the arm starts closed on the
            //    lower-right and opens diagonally as the shoulder opens slightly right, the elbow raises and turns
            //    right, and the forearm extends to grasp the left target.
            float forwardDist = Mathf.Lerp(0.45f, 0.49f, (float)rng.NextDouble());
            float lateralOffset = Mathf.Lerp(-0.015f, 0.015f, (float)rng.NextDouble());
            float planeY = GetPlaneElevationMeters(elevationPlane, lowPlaneHeight, midPlaneHeight, highPlaneHeight);
            float verticalJitter = Mathf.Lerp(-verticalJitterMax, verticalJitterMax, (float)rng.NextDouble());

            Vector3 startPos = new Vector3(
                baseOrigin.x - 0.185f + lateralOffset,
                baseOrigin.y + planeY + verticalJitter,
                baseOrigin.z + forwardDist
            );

            // 3. Final 6D transport target pose on the left in front of the body/head within reachable range,
            //    maintaining natural human wrist supination (palm inward toward body, thumb up)
            TargetElevationPlane targetPlane = (TargetElevationPlane)(((int)elevationPlane + 1 + rng.Next(2)) % 3);
            float targetPlaneY = GetPlaneElevationMeters(targetPlane, lowPlaneHeight, midPlaneHeight, highPlaneHeight);
            float targetHeight = Mathf.Clamp(targetPlaneY + 0.04f, minHeight, maxHeight);
            float targetForwardDist = Mathf.Lerp(0.46f, 0.50f, (float)rng.NextDouble());
            float targetLateralOffset = Mathf.Lerp(-0.015f, 0.015f, (float)rng.NextDouble());

            Vector3 targetPos = new Vector3(
                baseOrigin.x - 0.155f + targetLateralOffset,
                baseOrigin.y + targetHeight,
                baseOrigin.z + targetForwardDist
            );

            Quaternion targetRot = Quaternion.Euler(
                Mathf.Lerp(-12f, 10f, (float)rng.NextDouble()),
                startYawDeg + Mathf.Lerp(-15f, -5f, (float)rng.NextDouble()),
                Mathf.Lerp(-35f, -15f, (float)rng.NextDouble()) // Negative Roll: natural human supination!
            );

            return new SpawnedPayloadSpec
            {
                Shape = shape,
                ElevationPlane = elevationPlane,
                Mass = mass,
                Scale = scale,
                StartPosition = startPos,
                StartRotation = startRotation,
                TargetPosition = targetPos,
                TargetRotation = targetRot
            };
        }

        /// <summary>
        /// Builds a structured 3D room environment (floor + grid lines, back wall + horizon trim, left/right side walls,
        /// and a workspace table/pedestal under the manipulation zone) with distinct luminance levels and spatial edges
        /// so both human viewers and neuromorphic/optical vision sensors clearly distinguish floor from walls.
        /// </summary>
        public static Transform BuildStructuredRoomEnvironment(Transform parent, Vector3 baseOrigin, ArticulatedRobotRig rig = null)
        {
            Transform existing = parent != null ? parent.Find("StructuredWorkcellRoom") : null;
            if (existing != null)
            {
                existing.position = baseOrigin;
                existing.rotation = Quaternion.identity;
                IgnoreFloorCollisionsWithRig(existing, rig);
                return existing;
            }

            GameObject roomRoot = new GameObject("StructuredWorkcellRoom");
            if (parent != null)
            {
                roomRoot.transform.SetParent(parent, false);
            }
            roomRoot.transform.position = baseOrigin;
            roomRoot.transform.rotation = Quaternion.identity;

            Shader std = Shader.Find("Standard");

            Material CreateMat(string matName, Color col, float gloss = 0.15f, float metallic = 0.05f)
            {
                var m = new Material(std != null ? std : Shader.Find("Sprites/Default"))
                {
                    name = matName,
                    color = col
                };
                if (std != null)
                {
                    m.SetFloat("_Glossiness", gloss);
                    m.SetFloat("_Metallic", metallic);
                }
                return m;
            }

            // Distinct surface materials for clear floor-vs-wall visual contrast
            Material floorMat = CreateMat("RoomFloorMat", new Color(0.22f, 0.25f, 0.30f), 0.20f, 0.08f);
            Material floorGridMat = CreateMat("RoomFloorGridMat", new Color(0.32f, 0.37f, 0.44f), 0.15f, 0.05f);
            Material backWallMat = CreateMat("RoomBackWallMat", new Color(0.46f, 0.50f, 0.57f), 0.10f, 0.02f);
            Material sideWallMat = CreateMat("RoomSideWallMat", new Color(0.38f, 0.42f, 0.49f), 0.10f, 0.02f);
            Material baseboardMat = CreateMat("RoomBaseboardMat", new Color(0.12f, 0.14f, 0.18f), 0.25f, 0.15f);
            Material wainscotTrimMat = CreateMat("RoomWainscotTrimMat", new Color(0.28f, 0.33f, 0.40f), 0.20f, 0.10f);

            GameObject CreateBox(string boxName, Vector3 localPos, Vector3 localScale, Material mat, bool keepCollider = false)
            {
                GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cube);
                go.name = boxName;
                go.transform.SetParent(roomRoot.transform, false);
                go.transform.localPosition = localPos;
                go.transform.localRotation = Quaternion.identity;
                go.transform.localScale = localScale;
                if (!keepCollider)
                {
                    Collider c = go.GetComponent<Collider>();
                    if (c != null)
                    {
                        if (Application.isPlaying) Destroy(c);
                        else DestroyImmediate(c);
                    }
                }
                Renderer r = go.GetComponent<Renderer>();
                if (r != null)
                {
                    r.sharedMaterial = mat;
                    r.receiveShadows = true;
                }
                return go;
            }

            // 1. Main Floor Slab (y = 0 top surface, extending forward to z = +2.4m where it meets the back wall)
            CreateBox("IndustrialGroundFloor", new Vector3(0f, -0.02f, 0.60f), new Vector3(4.4f, 0.04f, 3.6f), floorMat, keepCollider: true);

            // Floor perspective grid strips (extending along Z and X on the floor to provide strong depth/perspective cues)
            for (int i = -3; i <= 3; i++)
            {
                float x = i * 0.45f;
                CreateBox($"FloorGridZ_{i}", new Vector3(x, 0.001f, 0.60f), new Vector3(0.018f, 0.002f, 3.5f), floorGridMat);
            }
            for (int k = 0; k <= 5; k++)
            {
                float z = -0.30f + k * 0.48f;
                CreateBox($"FloorGridX_{k}", new Vector3(0f, 0.001f, z), new Vector3(3.6f, 0.002f, 0.018f), floorGridMat);
            }

            // 2. Back Wall (at z = +2.35m, height = 3.2m) - distinctly lighter tone than the floor
            const float backWallZ = 2.35f;
            const float wallHeight = 3.2f;
            CreateBox("RoomBackWall", new Vector3(0f, 0.5f * wallHeight, backWallZ), new Vector3(4.4f, wallHeight, 0.06f), backWallMat);

            // High-contrast baseboard / skirting at floor-to-back-wall junction (y = 0.06m) so the floor/wall boundary is unmistakable
            CreateBox("BackWallBaseboard", new Vector3(0f, 0.06f, backWallZ - 0.035f), new Vector3(4.4f, 0.12f, 0.02f), baseboardMat);

            // Architectural horizontal wainscot / dado rail on the back wall (y = 1.05m and y = 1.60m) for visual horizon reference
            CreateBox("BackWallDadoRail", new Vector3(0f, 1.05f, backWallZ - 0.032f), new Vector3(4.4f, 0.035f, 0.015f), wainscotTrimMat);
            CreateBox("BackWallEyeRail", new Vector3(0f, 1.60f, backWallZ - 0.032f), new Vector3(4.4f, 0.025f, 0.015f), wainscotTrimMat);

            // Subtle vertical architectural pilasters on the back wall for lateral optical flow & parallax
            for (int i = -2; i <= 2; i++)
            {
                float x = i * 0.75f;
                CreateBox($"BackWallPilaster_{i}", new Vector3(x, 0.5f * wallHeight, backWallZ - 0.032f), new Vector3(0.03f, wallHeight, 0.015f), wainscotTrimMat);
            }

            // 3. Left & Right Side Walls (at x = -2.15m and x = +2.15m) with baseboards and dado rails
            const float sideWallX = 2.15f;
            CreateBox("RoomLeftWall", new Vector3(-sideWallX, 0.5f * wallHeight, 0.60f), new Vector3(0.06f, wallHeight, 3.6f), sideWallMat);
            CreateBox("LeftWallBaseboard", new Vector3(-sideWallX + 0.035f, 0.06f, 0.60f), new Vector3(0.02f, 0.12f, 3.6f), baseboardMat);
            CreateBox("LeftWallDadoRail", new Vector3(-sideWallX + 0.032f, 1.05f, 0.60f), new Vector3(0.015f, 0.035f, 3.6f), wainscotTrimMat);

            CreateBox("RoomRightWall", new Vector3(sideWallX, 0.5f * wallHeight, 0.60f), new Vector3(0.06f, wallHeight, 3.6f), sideWallMat);
            CreateBox("RightWallBaseboard", new Vector3(sideWallX - 0.035f, 0.06f, 0.60f), new Vector3(0.02f, 0.12f, 3.6f), baseboardMat);
            CreateBox("RightWallDadoRail", new Vector3(sideWallX - 0.032f, 1.05f, 0.60f), new Vector3(0.015f, 0.035f, 3.6f), wainscotTrimMat);

            IgnoreFloorCollisionsWithRig(roomRoot.transform, rig);
            return roomRoot.transform;
        }

        private static void IgnoreFloorCollisionsWithRig(Transform roomRoot, ArticulatedRobotRig rig)
        {
            if (roomRoot == null || rig == null) return;
            Collider[] roomColliders = roomRoot.GetComponentsInChildren<Collider>();
            Collider[] armColliders = rig.GetComponentsInChildren<Collider>();
            for (int r = 0; r < roomColliders.Length; r++)
            {
                for (int a = 0; a < armColliders.Length; a++)
                {
                    if (roomColliders[r] != null && armColliders[a] != null && roomColliders[r] != armColliders[a])
                    {
                        Physics.IgnoreCollision(roomColliders[r], armColliders[a], true);
                    }
                }
            }
        }

        private void EnsureGroundPlane(Vector3 baseOrigin, ArticulatedRobotRig rig)
        {
            if (_groundPlaneObject == null)
            {
                Transform room = BuildStructuredRoomEnvironment(transform, baseOrigin, rig);
                _groundPlaneObject = room.gameObject;
            }
            else
            {
                _groundPlaneObject.transform.position = baseOrigin;
                _groundPlaneObject.transform.rotation = Quaternion.identity;
                IgnoreFloorCollisionsWithRig(_groundPlaneObject.transform, rig);
            }
        }

        /// <summary>
        /// Spawns a new target payload directly in front of the head on the selected or randomized
        /// vertical elevation plane (Low, Mid, or High) for the episode seeded by episodeSeed.
        /// </summary>
        public SpawnedPayloadSpec SpawnEpisodePayload(int episodeSeed, ArticulatedRobotRig rig = null)
        {
            ClearActivePayload();

            _lastEpisodeSeed = episodeSeed;
            _lastRig = rig;

            Vector3 baseOrigin = rig != null ? rig.transform.position : transform.position;
            EnsureGroundPlane(baseOrigin, rig);

            TargetElevationPlane? forcedPlane = randomizeElevationPlane
                ? (TargetElevationPlane?)null
                : activeElevationPlane;

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
                maxStartAzimuth,
                forcedPlane,
                lowPlaneElevation,
                midPlaneElevation,
                highPlaneElevation,
                maxLateralJitter,
                planeVerticalJitter
            );

            activeElevationPlane = _currentSpec.ElevationPlane;

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
            _activePayloadObject.name = $"TargetPayload_{_currentSpec.ElevationPlane}_{_currentSpec.Shape}_{episodeSeed:D4}";
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
            // Hold target stationary at its spawned vertical plane (Low, Mid, or High) until gripper contact;
            // PayloadGraspConstraint.AttachRigidly enables gravity and releases constraints on contact so full mass/inertia pull on the arm.
            _activePayloadRigidbody.useGravity = false;
            _activePayloadRigidbody.constraints = RigidbodyConstraints.FreezeAll;
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
