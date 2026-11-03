using System.Collections.Generic;
using UnityEngine;
using SPWM.Robotics;

namespace SPWM.Visualization
{
    /// <summary>
    /// Purely kinematic (physics-free) replica of <see cref="ArticulatedRobotRig"/> for playback.
    /// Built from the same <see cref="ArticulatedRobotRig.GetArchetypeSpecs"/> table, so joint order
    /// and link segment geometry match the bolted 6-DOF industrial robot arm exactly.
    /// Each link = unscaled pivot transform (rotates around the exact physical pivot) + scaled visual shaft
    /// spanning [0, d.y] + bolted spherical joint housing at (0, 0, 0) + magnetic gripper tip on the wrist roll link.
    /// </summary>
    public sealed class KinematicRobotRig
    {
        private readonly List<Transform> _jointPivots = new List<Transform>();
        private readonly List<Vector3> _jointAxes = new List<Vector3>();
        private readonly List<Transform> _links = new List<Transform>();
        private readonly List<float> _linkMasses = new List<float>();
        private readonly List<Renderer> _renderers = new List<Renderer>();
        private float _totalMass;

        public Transform Root { get; }
        public Transform EndEffectorTip { get; private set; }
        public Transform WristCameraMount { get; private set; }
        public int JointCount => _jointPivots.Count;
        public IReadOnlyList<Renderer> Renderers => _renderers;

        public KinematicRobotRig(string name, Transform parent, RobotArchetype archetype, Material material,
                                 Vector3 localOffset, bool forceSagittalAxis)
        {
            Root = new GameObject(name).transform;
            Root.SetParent(parent, false);
            Root.localPosition = localOffset;
            Root.localRotation = Quaternion.identity;
            Root.localScale = Vector3.one;

            List<ArticulatedLinkSpec> specs = ArticulatedRobotRig.GetArchetypeSpecs(archetype);
            var created = new List<Transform>(specs.Count);

            for (int i = 0; i < specs.Count; i++)
            {
                ArticulatedLinkSpec spec = specs[i];
                var pivot = new GameObject(spec.Name).transform;
                pivot.SetParent(spec.ParentIndex < 0 ? Root : created[spec.ParentIndex], false);
                pivot.localPosition = spec.LocalAnchorPosition;
                pivot.localRotation = Quaternion.identity;
                pivot.localScale = Vector3.one;

                bool isRoot = spec.ParentIndex < 0;
                bool isEndEffector = i == specs.Count - 1;
                CreateVisual(pivot, spec, material, isRoot, isEndEffector);
                created.Add(pivot);
                _links.Add(pivot);
                _linkMasses.Add(Mathf.Max(0.01f, spec.Mass));
                _totalMass += Mathf.Max(0.01f, spec.Mass);

                if (!isRoot)
                {
                    // ArticulationBody revolute joints rotate about the anchor frame's X axis.
                    Vector3 axis = forceSagittalAxis
                        ? Vector3.right
                        : (Quaternion.Euler(spec.AxisRotationEuler) * Vector3.right).normalized;
                    _jointPivots.Add(pivot);
                    _jointAxes.Add(axis);
                }
            }
        }

        private static void SafeDestroy(Object obj)
        {
            if (obj == null) return;
            if (Application.isPlaying) Object.Destroy(obj);
            else Object.DestroyImmediate(obj);
        }

        private void CreateVisual(Transform pivot, ArticulatedLinkSpec spec, Material material, bool isRoot, bool isEndEffector)
        {
            Vector3 d = spec.LinkDimensions;

            if (isRoot)
            {
                // Root socket sits silently at the right shoulder mount without protruding torso/clavicle blocks into the camera
                return;
            }

            // Anatomical spherical joint articulation (deltoid shoulder cap, elbow, wrist) centered at (0, 0, 0).
            // Shoulder_Pan is kept minimal as a base socket.
            if (spec.Name != "Shoulder_Pan")
            {
                GameObject hub = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                hub.name = spec.Name + "_JointHub";
                SafeDestroy(hub.GetComponent<Collider>());
                hub.transform.SetParent(pivot, false);
                float hubDiameter = Mathf.Max(d.x, d.z) * (spec.Name.StartsWith("Wrist") ? 1.02f : 1.06f);
                hub.transform.localScale = new Vector3(hubDiameter, hubDiameter, hubDiameter);
                hub.transform.localPosition = Vector3.zero;
                RegisterRenderer(hub.GetComponent<Renderer>(), material);
            }

            // Rigid cylindrical structural limb segment.
            // Each link renders its own anatomical collar/shaft spanning (0, 0, 0) -> (0, d.y, 0).
            if (spec.Name == "Wrist_Yaw")
            {
                // Intermediate kinematic anchor bridging Wrist_Pitch and Wrist_Roll
                GameObject yawCollar = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                yawCollar.name = "Wrist_Yaw_Visual";
                SafeDestroy(yawCollar.GetComponent<Collider>());
                yawCollar.transform.SetParent(pivot, false);
                yawCollar.transform.localScale = new Vector3(d.x, 0.5f * d.y, d.z);
                yawCollar.transform.localPosition = new Vector3(0f, 0.5f * d.y, 0f);
                yawCollar.transform.localRotation = Quaternion.identity;
                RegisterRenderer(yawCollar.GetComponent<Renderer>(), material);
            }
            else if (spec.Name == "Wrist_Pitch")
            {
                // Anatomical wrist cuff segment
                GameObject wristSleeve = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                wristSleeve.name = "Wrist_Sleeve_Visual";
                SafeDestroy(wristSleeve.GetComponent<Collider>());
                wristSleeve.transform.SetParent(pivot, false);
                wristSleeve.transform.localScale = new Vector3(d.x, 0.5f * d.y, d.z);
                wristSleeve.transform.localPosition = new Vector3(0f, 0.5f * d.y, 0f);
                wristSleeve.transform.localRotation = Quaternion.identity;
                RegisterRenderer(wristSleeve.GetComponent<Renderer>(), material);
            }
            else if (spec.Name == "Wrist_Roll")
            {
                // Terminal wrist rotation link before the hand; renders a solid wrist band collar bridging directly to the hand
                GameObject wristBand = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                wristBand.name = "Wrist_Band_Visual";
                SafeDestroy(wristBand.GetComponent<Collider>());
                wristBand.transform.SetParent(pivot, false);
                wristBand.transform.localScale = new Vector3(d.x, 0.5f * d.y, d.z);
                wristBand.transform.localPosition = new Vector3(0f, 0.5f * d.y, 0f);
                wristBand.transform.localRotation = Quaternion.identity;
                RegisterRenderer(wristBand.GetComponent<Renderer>(), material);
            }
            else
            {
                GameObject shaft = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                shaft.name = spec.Name + "_VisualShaft";
                SafeDestroy(shaft.GetComponent<Collider>());
                shaft.transform.SetParent(pivot, false);
                shaft.transform.localScale = new Vector3(d.x, 0.5f * d.y, d.z);
                shaft.transform.localPosition = new Vector3(0f, 0.5f * d.y, 0f);
                shaft.transform.localRotation = Quaternion.identity;
                RegisterRenderer(shaft.GetComponent<Renderer>(), material);
            }

            if (isEndEffector)
            {
                // Humanoid right-hand palm and fingers anchored at the distal end of Wrist_Roll (0, d.y, 0)
                var tip = new GameObject("EndEffectorTip").transform;
                tip.SetParent(pivot, false);
                tip.localPosition = new Vector3(0f, d.y, 0f);
                tip.localRotation = Quaternion.identity;
                tip.localScale = Vector3.one;
                EndEffectorTip = tip;

                // Egocentric wrist camera mount on the dorsal side (-Z) of the wrist looking forward (+Y) over the hand
                var wristMount = new GameObject("WristCameraMount").transform;
                wristMount.SetParent(tip, false);
                wristMount.localPosition = new Vector3(0f, -0.015f, -0.042f);
                wristMount.localRotation = Quaternion.Euler(-82f, 0f, 0f);
                wristMount.localScale = Vector3.one;
                WristCameraMount = wristMount;

                // Humanoid right-hand palm (metacarpals): wide across X (7.6cm), extending 5.2cm along +Y, thin across Z (2.2cm)
                GameObject palm = GameObject.CreatePrimitive(PrimitiveType.Cube);
                palm.name = "HumanoidPalm";
                SafeDestroy(palm.GetComponent<Collider>());
                palm.transform.SetParent(tip, false);
                palm.transform.localPosition = new Vector3(0f, 0.012f, 0.004f);
                palm.transform.localRotation = Quaternion.identity;
                palm.transform.localScale = new Vector3(0.076f, 0.052f, 0.022f);
                RegisterRenderer(palm.GetComponent<Renderer>(), material);

                // Knuckle bridge across the distal end of the palm
                GameObject knuckles = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                knuckles.name = "HumanoidKnuckles";
                SafeDestroy(knuckles.GetComponent<Collider>());
                knuckles.transform.SetParent(tip, false);
                knuckles.transform.localPosition = new Vector3(0f, 0.038f, 0.004f);
                knuckles.transform.localRotation = Quaternion.Euler(0f, 0f, 90f);
                knuckles.transform.localScale = new Vector3(0.024f, 0.039f, 0.024f);
                RegisterRenderer(knuckles.GetComponent<Renderer>(), material);

                // 4 Articulated fingers (Index, Middle, Ring, Little) with proximal + distal phalanges curving around the grasp zone
                float[] fingerOffsetsX = new float[] { -0.027f, -0.009f, 0.009f, 0.027f };
                float[] proxLengths = new float[] { 0.034f, 0.038f, 0.036f, 0.029f };
                float[] distLengths = new float[] { 0.026f, 0.029f, 0.027f, 0.022f };
                for (int f = 0; f < fingerOffsetsX.Length; f++)
                {
                    GameObject prox = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                    prox.name = $"HumanoidFingerProx_{f}";
                    SafeDestroy(prox.GetComponent<Collider>());
                    prox.transform.SetParent(tip, false);
                    Quaternion proxRot = Quaternion.Euler(18f, 0f, 0f);
                    Vector3 proxDir = proxRot * Vector3.up;
                    Vector3 knucklePos = new Vector3(fingerOffsetsX[f], 0.038f, 0.004f);
                    prox.transform.localPosition = knucklePos + proxDir * (0.5f * proxLengths[f]);
                    prox.transform.localRotation = proxRot;
                    prox.transform.localScale = new Vector3(0.015f, 0.5f * proxLengths[f], 0.015f);
                    RegisterRenderer(prox.GetComponent<Renderer>(), material);

                    Vector3 pipPos = knucklePos + proxDir * proxLengths[f];
                    GameObject dist = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                    dist.name = $"HumanoidFingerDist_{f}";
                    SafeDestroy(dist.GetComponent<Collider>());
                    dist.transform.SetParent(tip, false);
                    Quaternion distRot = Quaternion.Euler(48f, 0f, 0f);
                    Vector3 distDir = distRot * Vector3.up;
                    dist.transform.localPosition = pipPos + distDir * (0.5f * distLengths[f]);
                    dist.transform.localRotation = distRot;
                    dist.transform.localScale = new Vector3(0.013f, 0.5f * distLengths[f], 0.013f);
                    RegisterRenderer(dist.GetComponent<Renderer>(), material);
                }

                // Right-hand thumb on the radial/medial side (-X) opposing the fingers
                // Also serves as the asymmetric visual orientation fiducial so Wrist_Roll (q5) is optically distinct
                GameObject thumbMeta = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                thumbMeta.name = "WristOrientationFiducial";
                SafeDestroy(thumbMeta.GetComponent<Collider>());
                thumbMeta.transform.SetParent(tip, false);
                thumbMeta.transform.localPosition = new Vector3(-0.044f, 0.014f, 0.010f);
                thumbMeta.transform.localRotation = Quaternion.Euler(22f, -15f, 38f);
                thumbMeta.transform.localScale = new Vector3(0.018f, 0.024f, 0.018f);
                RegisterRenderer(thumbMeta.GetComponent<Renderer>(), material);

                GameObject thumbDist = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                thumbDist.name = "HumanoidThumbDistal";
                SafeDestroy(thumbDist.GetComponent<Collider>());
                thumbDist.transform.SetParent(tip, false);
                thumbDist.transform.localPosition = new Vector3(-0.052f, 0.042f, 0.022f);
                thumbDist.transform.localRotation = Quaternion.Euler(35f, 15f, 12f);
                thumbDist.transform.localScale = new Vector3(0.016f, 0.020f, 0.016f);
                RegisterRenderer(thumbDist.GetComponent<Renderer>(), material);
            }
        }

        private void RegisterRenderer(Renderer r, Material material)
        {
            if (r == null) return;
            r.sharedMaterial = material;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
            _renderers.Add(r);
        }

        /// <summary>Precomputes the local rotation for joint j at angle q (radians).</summary>
        public Quaternion JointRotation(int j, float qRad, float angleScale)
        {
            return Quaternion.AngleAxis(qRad * Mathf.Rad2Deg * angleScale, _jointAxes[j]);
        }

        public void SetJointLocalRotation(int j, Quaternion rotation) => _jointPivots[j].localRotation = rotation;

        public void SetShadowCasting(bool on)
        {
            var mode = on ? UnityEngine.Rendering.ShadowCastingMode.On : UnityEngine.Rendering.ShadowCastingMode.Off;
            foreach (Renderer r in _renderers) { r.shadowCastingMode = mode; r.receiveShadows = on; }
        }

        /// <summary>Mass-weighted world-space centre of the link pivots.</summary>
        public Vector3 CenterOfMass()
        {
            Vector3 acc = Vector3.zero;
            for (int i = 0; i < _links.Count; i++) acc += _links[i].position * _linkMasses[i];
            return _totalMass > 0f ? acc / _totalMass : Root.position;
        }

        public Bounds WorldBounds()
        {
            if (_renderers.Count == 0) return new Bounds(Root.position, Vector3.one);
            Bounds b = _renderers[0].bounds;
            for (int i = 1; i < _renderers.Count; i++) b.Encapsulate(_renderers[i].bounds);
            return b;
        }
    }

    /// <summary>
    /// Ghost execution playback: instantiates an opaque Target 6-DOF industrial robot rig driven by ground_truth_q,
    /// a holographic Ghost rig driven by predicted_q, the bolted industrial floor, and the heavy target payload object
    /// sitting on the ground, being grabbed at the grasp frame, lifted sharply during Heft, and carried + twisted
    /// upright during Transport.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class GhostPlaybackManager : MonoBehaviour
    {
        [Header("Data")]
        [SerializeField] private string trajectoryFileName = "trajectory_ghost.json";

        [Header("Rig")]
        [SerializeField] private RobotArchetype archetype = RobotArchetype.ManipulatorArm;
        [Tooltip("Matte, opaque material for the ground-truth (target) robot.")]
        [SerializeField] private Material targetMaterial;
        [Tooltip("HolographicGhost material (SPWM/HolographicGhost shader) for the predicted robot.")]
        [SerializeField] private Material ghostMaterial;
        [Tooltip("World offset of the ghost relative to the target. Zero = perfect overlay (recommended).")]
        [SerializeField] private Vector3 ghostOffset = Vector3.zero;
        [Tooltip("Rotate every joint about local X (sagittal pitch) instead of the sim's anchor axis.")]
        [SerializeField] private bool forceSagittalAxis = false;
        [Tooltip("Visual exaggeration of joint angles. Keep 1 for physically faithful playback.")]
        [SerializeField, Range(0.25f, 3f)] private float angleScale = 1f;

        [Header("Grasp & Payload Visualization (Anthropomorphic 3-Plane Spawning)")]
        [Tooltip("Playback frame index (0-indexed) at which the end-effector locks onto the target object (start of Heft).")]
        [SerializeField] private int graspLockFrameIndex = 35;
        [SerializeField] private Vector3 payloadDimensions = new Vector3(0.11f, 0.07f, 0.09f);
        [Tooltip("If true, randomizes target spawn elevation among Low, Mid, and High planes based on trajectory seed. If false, uses activeElevationPlane.")]
        [SerializeField] private bool randomizeTargetElevation = true;
        [Tooltip("Active vertical elevation plane in front of the head (Low, Mid, or High). Press 'E' during playback to toggle.")]
        [SerializeField] private TargetElevationPlane activeElevationPlane = TargetElevationPlane.Mid;
        [SerializeField] private float lowPlaneElevation = ProceduralObjectSpawner.DefaultLowPlaneHeight;
        [SerializeField] private float midPlaneElevation = ProceduralObjectSpawner.DefaultMidPlaneHeight;
        [SerializeField] private float highPlaneElevation = ProceduralObjectSpawner.DefaultHighPlaneHeight;

        [Header("Playback")]
        [SerializeField] private bool playOnStart = true;
        [SerializeField, Min(0f)] private float startDelaySeconds = 1.0f;
        [Tooltip("1 = real time (100 steps x 50 ms = 5 s). Use 0.5 for slow-motion social clips.")]
        [SerializeField, Range(0.05f, 4f)] private float playbackSpeed = 0.5f;
        [SerializeField] private bool loop = true;
        [SerializeField, Min(0f)] private float holdAtEndSeconds = 2.0f;

        private GhostTrajectoryData _data;
        private KinematicRobotRig _target;
        private KinematicRobotRig _ghost;
        private Quaternion[,] _targetRot; // [frame, joint] precomputed
        private Quaternion[,] _ghostRot;
        private Transform _groundVisual;
        private Transform _payloadVisual;
        private Vector3 _payloadGroundPosition;
        private Quaternion _payloadGroundRotation;
        private Vector3 _payloadGraspLocalPos;
        private Quaternion _payloadGraspLocalRot;
        private SpawnedPayloadSpec _activePayloadSpec;
        private bool _hasManualElevationOverride;
        private float _playbackTime;      // seconds of simulated time since frame 0
        private float _delayRemaining;
        private float _holdRemaining;
        private bool _playing;

        public GhostTrajectoryData Data => _data;
        public bool IsLoaded => _data != null;
        public bool IsPlaying => _playing;
        public KinematicRobotRig TargetRig => _target;
        public KinematicRobotRig GhostRig => _ghost;
        public int FrameCount => _data != null ? _data.FrameCount : 0;
        public TargetElevationPlane ActiveElevationPlane => activeElevationPlane;
        public Vector3 InitialGraspTargetWorldPosition => _payloadGroundPosition;

        /// <summary>
        /// Toggles sequentially between Low -> Mid -> High vertical target elevation planes directly in front of the head.
        /// </summary>
        [ContextMenu("Toggle Target Elevation Plane (Low / Mid / High)")]
        public TargetElevationPlane ToggleTargetElevationPlane()
        {
            randomizeTargetElevation = false;
            _hasManualElevationOverride = true;
            activeElevationPlane = (TargetElevationPlane)(((int)activeElevationPlane + 1) % 3);
            if (_data != null && _target != null && _ghost != null)
            {
                PrecomputeRotations();
                InitializeEnvironmentAndPayload();
                ApplyPose(FramePosition);
            }
            return activeElevationPlane;
        }

        /// <summary>
        /// Explicitly sets the target elevation plane (Low, Mid, or High) in front of the head.
        /// </summary>
        public void SetTargetElevationPlane(TargetElevationPlane plane, bool disableRandomization = true)
        {
            activeElevationPlane = plane;
            if (disableRandomization)
            {
                randomizeTargetElevation = false;
                _hasManualElevationOverride = true;
            }
            if (_data != null && _target != null && _ghost != null)
            {
                PrecomputeRotations();
                InitializeEnvironmentAndPayload();
                ApplyPose(FramePosition);
            }
        }

        public void SetGhostVisible(bool visible)
        {
            if (_ghost != null)
            {
                foreach (Renderer r in _ghost.Renderers)
                {
                    if (r != null) r.enabled = visible;
                }
            }
        }

        /// <summary>Continuous frame index in [0, FrameCount-1]; fractional part = interpolation alpha.</summary>
        public float FramePosition { get; private set; }

        public int GraspLockFrame => graspLockFrameIndex;
        public int HeftStartFrame => _data != null && _data.metadata != null ? _data.metadata.heft_start_frame_index : -1;
        public int SensoryReleaseFrame => _data != null && _data.metadata != null ? _data.metadata.sensory_release_frame_index : -1;

        /// <summary>1-indexed horizon step currently displayed (t in "t / H").</summary>
        public int CurrentStep => _data == null ? 0 : Mathf.Clamp(Mathf.FloorToInt(FramePosition) + 1, 1, FrameCount);

        /// <summary>Normalised progress through the horizon in [0, 1].</summary>
        public float Progress => FrameCount > 1 ? FramePosition / (FrameCount - 1) : 0f;

        public event System.Action<GhostPlaybackManager> Loaded;
        public event System.Action<GhostPlaybackManager> PlaybackRestarted;

        public void LoadTrajectory(string filenameOrPath)
        {
            if (System.IO.File.Exists(filenameOrPath))
            {
                string json;
                if (filenameOrPath.EndsWith(".gz", System.StringComparison.OrdinalIgnoreCase))
                {
                    using (var fileStream = System.IO.File.OpenRead(filenameOrPath))
                    using (var gzipStream = new System.IO.Compression.GZipStream(fileStream, System.IO.Compression.CompressionMode.Decompress))
                    using (var reader = new System.IO.StreamReader(gzipStream, System.Text.Encoding.UTF8))
                    {
                        json = reader.ReadToEnd();
                    }
                }
                else
                {
                    json = System.IO.File.ReadAllText(filenameOrPath);
                }
                _data = GhostTrajectoryData.FromJson(json);
            }
            else
            {
                _data = GhostTrajectoryData.LoadFromStreamingAssets(filenameOrPath);
            }

            EnsureMaterials();
            if (_target == null)
            {
                _target = new KinematicRobotRig("TargetRig_GroundTruth", transform, archetype, targetMaterial, Vector3.zero, forceSagittalAxis);
            }
            if (_ghost == null)
            {
                _ghost = new KinematicRobotRig("GhostRig_SPWM", transform, archetype, ghostMaterial, ghostOffset, forceSagittalAxis);
                _ghost.SetShadowCasting(false);
            }

            if (_target.JointCount != _data.JointCount)
            {
                Debug.LogError($"[GhostPlaybackManager] Rig has {_target.JointCount} actuated joints but data has " +
                               $"{_data.JointCount}. Check archetype.", this);
                enabled = false;
                return;
            }

            if (_data.metadata != null && _data.metadata.grasp_lock_frame_index >= 0)
            {
                graspLockFrameIndex = _data.metadata.grasp_lock_frame_index;
            }
            PrecomputeRotations();
            InitializeEnvironmentAndPayload();
            ApplyPose(0f);
            Loaded?.Invoke(this);
        }

        private void Awake()
        {
            try
            {
                LoadTrajectory(trajectoryFileName);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[GhostPlaybackManager] {e.Message}", this);
                enabled = false;
                return;
            }
        }

        private void Start()
        {
            if (enabled && playOnStart) Restart();
        }

        private void EnsureMaterials()
        {
            if (targetMaterial == null)
            {
                targetMaterial = new Material(Shader.Find("Standard")) { name = "GroundTruthMatte (runtime)" };
                targetMaterial.color = new Color(0.78f, 0.80f, 0.84f);
                targetMaterial.SetFloat("_Glossiness", 0.22f);
                targetMaterial.SetFloat("_Metallic", 0.15f);
            }
            if (ghostMaterial == null)
            {
                Shader holo = Shader.Find("SPWM/HolographicGhost");
                if (holo == null)
                {
                    Debug.LogWarning("[GhostPlaybackManager] SPWM/HolographicGhost shader not found; using Unlit/Color.", this);
                    holo = Shader.Find("Unlit/Color");
                }
                ghostMaterial = new Material(holo) { name = "HolographicGhost (runtime)" };
            }
        }

        private static float SmoothStepMinimumJerk(float tau)
        {
            float t = Mathf.Clamp01(tau);
            return t * t * t * (10.0f + t * (-15.0f + 6.0f * t));
        }

        private void PrecomputeRotations()
        {
            int frames = _data.FrameCount, joints = _data.JointCount;
            _targetRot = new Quaternion[frames, joints];
            _ghostRot = new Quaternion[frames, joints];

            int seed = (_data.metadata != null && _data.metadata.trajectory_id > 0)
                ? _data.metadata.trajectory_id
                : 42;
            TargetElevationPlane? forcedPlane = randomizeTargetElevation
                ? (TargetElevationPlane?)null
                : activeElevationPlane;

            _activePayloadSpec = ProceduralObjectSpawner.GenerateDeterministicSpec(
                episodeSeed: seed,
                baseOrigin: transform.position,
                forcedElevationPlane: forcedPlane,
                lowPlaneHeight: lowPlaneElevation,
                midPlaneHeight: midPlaneElevation,
                highPlaneHeight: highPlaneElevation
            );
            activeElevationPlane = _activePayloadSpec.ElevationPlane;
            payloadDimensions = _activePayloadSpec.Scale;

            // Detect if the loaded JSON was exported with legacy/clamped joint angles (where q1 < 2.65 rad or |q2| < 1.20 rad),
            // or if the user manually toggled the target elevation plane (Low / Mid / High)
            bool adaptToAnthropomorphicPose = _hasManualElevationOverride ||
                (frames > 0 && _data.frames[0].ground_truth_q != null &&
                 _data.frames[0].ground_truth_q.Length > 2 &&
                 (_data.frames[0].ground_truth_q[1] < 2.65f || Mathf.Abs(_data.frames[0].ground_truth_q[2]) < 1.20f));

            float[] qNeutral = ArticulatedRobotRig.GetNeutralRestingPose();
            Vector3 graspPalm = ArticulatedRobotRig.PalmNormalFromFrontalAngleDeg(270f);
            Vector3 finalPalm = ArticulatedRobotRig.PalmNormalFromFrontalAngleDeg(135f);
            float[] qGrasp = ArticulatedRobotRig.SolveInverseKinematicsOrientedStatic(
                transform.position, _activePayloadSpec.GraspPoint, graspPalm);
            Vector3 liftedPoint = _activePayloadSpec.GraspPoint + new Vector3(0f, 0.10f, 0f);
            float[] qHeftPeak = ArticulatedRobotRig.SolveInverseKinematicsOrientedStatic(transform.position, liftedPoint, graspPalm);
            float[] qTarget6D = ArticulatedRobotRig.SolveInverseKinematicsOrientedStatic(
                transform.position, _activePayloadSpec.TargetPosition, finalPalm);

            if (joints >= 6)
            {
                qHeftPeak[3] = Mathf.Clamp(qHeftPeak[3] + 0.08f, -2.60f, 2.60f);
                qHeftPeak[4] = Mathf.Clamp(qHeftPeak[4] - 0.05f, -1.25f, 1.25f);
                qHeftPeak[5] = Mathf.Clamp(qHeftPeak[5] - 0.20f, -ArticulatedRobotRig.MaxJointAngleRad, ArticulatedRobotRig.MaxJointAngleRad);
            }

            int setupHold = 4;
            int lockFrame = Mathf.Clamp(graspLockFrameIndex, setupHold + 4, Mathf.Max(setupHold + 5, frames - 20));
            int heftEnd = Mathf.Min(frames - 5, lockFrame + 12);

            for (int f = 0; f < frames; f++)
            {
                GhostFrame fr = _data.frames[f];
                // Smoothly fade in prediction error over initial warm-up steps so rollout starts clean
                float warmupBlend = SmoothStepMinimumJerk((float)f / (float)setupHold);

                for (int j = 0; j < joints; j++)
                {
                    float gtQ = fr.ground_truth_q[j];
                    float predQ = fr.predicted_q[j];

                    if (adaptToAnthropomorphicPose && j < CanonicalPoseLength(qNeutral))
                    {
                        float errDelta = (predQ - gtQ) * warmupBlend;
                        if (j == 5)
                        {
                            // Bound prediction error on wrist roll so ghost model stays anatomically grounded
                            errDelta = Mathf.Clamp(errDelta, -0.15f, 0.15f);
                        }
                        float anthroQ;
                        if (f < setupHold)
                        {
                            anthroQ = qNeutral[j];
                        }
                        else if (f < lockFrame)
                        {
                            float s = SmoothStepMinimumJerk((float)(f - setupHold + 1) / Mathf.Max(1, lockFrame - setupHold));
                            anthroQ = Mathf.Lerp(qNeutral[j], qGrasp[j], s);
                        }
                        else if (f < heftEnd)
                        {
                            float s = SmoothStepMinimumJerk((float)(f - lockFrame) / Mathf.Max(1, heftEnd - lockFrame));
                            anthroQ = Mathf.Lerp(qGrasp[j], qHeftPeak[j], s);
                        }
                        else
                        {
                            float s = SmoothStepMinimumJerk((float)(f - heftEnd + 1) / Mathf.Max(1, frames - heftEnd));
                            anthroQ = Mathf.Lerp(qHeftPeak[j], qTarget6D[j], s);
                        }
                        gtQ = anthroQ;
                        predQ = Mathf.Clamp(anthroQ + errDelta, -ArticulatedRobotRig.MaxJointAngleRad, ArticulatedRobotRig.MaxJointAngleRad);
                    }
                    else
                    {
                        // Clean start grounding: blend raw prediction error from 0 to 1 over warmup steps
                        predQ = gtQ + (predQ - gtQ) * warmupBlend;
                    }

                    _targetRot[f, j] = _target.JointRotation(j, gtQ, angleScale);
                    _ghostRot[f, j] = _ghost.JointRotation(j, predQ, angleScale);
                }
            }
        }

        private static int CanonicalPoseLength(float[] arr) => arr != null ? arr.Length : 0;

        /// <summary>
        /// Instantiates the industrial ground plane and the target payload placed directly in front of the head
        /// on the active vertical elevation plane (Low, Mid, or High), so the horizontally extended arm approaches it,
        /// locks onto it, performs a sharp upward Heft lift, and carries + twists it during Transport.
        /// </summary>
        private static void SafeDestroy(Object obj)
        {
            if (obj == null) return;
            if (Application.isPlaying) Object.Destroy(obj);
            else Object.DestroyImmediate(obj);
        }

        private void InitializeEnvironmentAndPayload()
        {
            // 1. Structured 3D workcell room with distinct floor, grid lines, baseboards, and walls for clear visual perception
            if (_groundVisual == null)
            {
                _groundVisual = ProceduralObjectSpawner.BuildStructuredRoomEnvironment(transform, transform.position, rig: null);
            }
            else
            {
                _groundVisual.position = transform.position;
                _groundVisual.rotation = Quaternion.identity;
            }

            // 2. Pose the target rig at graspLockFrameIndex to read the exact world-space gripper tip pose on contact
            int lockFrame = Mathf.Clamp(graspLockFrameIndex, 0, _data.FrameCount - 1);
            for (int j = 0; j < _data.JointCount; j++)
            {
                _target.SetJointLocalRotation(j, _targetRot[lockFrame, j]);
            }

            Transform tip = _target.EndEffectorTip;
            Vector3 headPos = ArticulatedRobotRig.GetAnthropomorphicHeadPosition(transform.position);
            float defaultPlaneY = ProceduralObjectSpawner.GetPlaneElevationMeters(
                activeElevationPlane, lowPlaneElevation, midPlaneElevation, highPlaneElevation);
            Vector3 tipWorldPos = tip != null ? tip.position : new Vector3(headPos.x, transform.position.y + defaultPlaneY, transform.position.z + 0.48f);
            Quaternion tipWorldRot = tip != null ? tip.rotation : Quaternion.Euler(90f, 0f, 0f);

            // Place the target payload cradled right inside the humanoid palm & curved fingers
            // (local +Y = forward along hand, local +Z = downward/palmar side where fingers curl)
            float halfDepth = 0.5f * payloadDimensions.z;
            _payloadGroundPosition = tipWorldPos + tipWorldRot * new Vector3(0f, 0.035f + 0.65f * halfDepth, 0.024f);
            _payloadGroundRotation = _activePayloadSpec.StartRotation;

            // Compute exact rigid-lock offset relative to the gripper tip at the moment of magnetic lock
            _payloadGraspLocalPos = Quaternion.Inverse(tipWorldRot) * (_payloadGroundPosition - tipWorldPos);
            _payloadGraspLocalRot = Quaternion.Inverse(tipWorldRot) * _payloadGroundRotation;

            // 3. Instantiate or update the visual target payload block
            if (_payloadVisual == null)
            {
                GameObject payloadObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
                SafeDestroy(payloadObj.GetComponent<Collider>());
                payloadObj.transform.SetParent(transform, true);
                _payloadVisual = payloadObj.transform;

                var payloadRenderer = payloadObj.GetComponent<Renderer>();
                if (payloadRenderer != null)
                {
                    Shader s = Shader.Find("Standard");
                    if (s != null)
                    {
                        var payloadMat = new Material(s)
                        {
                            name = "TargetPayloadMat (runtime)",
                            color = new Color(0.92f, 0.44f, 0.14f)
                        };
                        payloadMat.SetFloat("_Glossiness", 0.38f);
                        payloadMat.SetFloat("_Metallic", 0.25f);
                        payloadRenderer.sharedMaterial = payloadMat;
                    }
                }
            }

            _payloadVisual.name = $"TargetPayload_{activeElevationPlane}Plane_Block";
            _payloadVisual.localScale = payloadDimensions;
            _payloadVisual.SetPositionAndRotation(_payloadGroundPosition, _payloadGroundRotation);
        }

        public void Restart()
        {
            _playbackTime = 0f;
            _delayRemaining = startDelaySeconds;
            _holdRemaining = holdAtEndSeconds;
            _playing = true;
            ApplyPose(0f);
            PlaybackRestarted?.Invoke(this);
        }

        public void TogglePause() => _playing = !_playing;

        private void Update()
        {
            HandleHotkeys();
            if (!_playing || _data == null) return;

            if (_delayRemaining > 0f) { _delayRemaining -= Time.deltaTime; return; }

            float duration = (_data.FrameCount - 1) * _data.Dt;
            if (_playbackTime < duration)
            {
                _playbackTime = Mathf.Min(duration, _playbackTime + Time.deltaTime * playbackSpeed);
            }
            else if ((_holdRemaining -= Time.deltaTime) <= 0f)
            {
                if (loop) Restart(); else _playing = false;
                return;
            }

            ApplyPose(_playbackTime / _data.Dt);
        }

        /// <summary>
        /// Poses both rigs at continuous frame position <paramref name="framePos"/>: frames i and i+1 are
        /// blended with Quaternion.Slerp using alpha = frac(framePos), giving smooth motion between 50 ms steps.
        /// Updates the target payload pose: sitting on the ground before graspLockFrameIndex, and rigidly
        /// locked to the end-effector tip during Heft and Transport.
        /// </summary>
        private void ApplyPose(float framePos)
        {
            int last = _data.FrameCount - 1;
            FramePosition = Mathf.Clamp(framePos, 0f, last);
            int i0 = Mathf.FloorToInt(FramePosition);
            int i1 = Mathf.Min(i0 + 1, last);
            float alpha = FramePosition - i0;

            for (int j = 0; j < _data.JointCount; j++)
            {
                _target.SetJointLocalRotation(j, Quaternion.Slerp(_targetRot[i0, j], _targetRot[i1, j], alpha));
                _ghost.SetJointLocalRotation(j, Quaternion.Slerp(_ghostRot[i0, j], _ghostRot[i1, j], alpha));
            }

            if (_payloadVisual != null)
            {
                if (FramePosition < graspLockFrameIndex || _target.EndEffectorTip == null)
                {
                    // Before Heft: payload sits stationary on the ground waiting for the arm's approach
                    _payloadVisual.SetPositionAndRotation(_payloadGroundPosition, _payloadGroundRotation);
                }
                else
                {
                    // Heft & Transport: payload is rigidly locked to the magnetic gripper tip
                    Transform tip = _target.EndEffectorTip;
                    Vector3 lockedPos = tip.position + tip.rotation * _payloadGraspLocalPos;
                    Quaternion lockedRot = tip.rotation * _payloadGraspLocalRot;
                    _payloadVisual.SetPositionAndRotation(lockedPos, lockedRot);
                }
            }
        }

        /// <summary>Linearly interpolated instantaneous metrics at the current frame position.</summary>
        public GhostInstantaneousMetrics SampleMetrics()
        {
            var m = new GhostInstantaneousMetrics();
            if (_data == null) return m;
            int last = _data.FrameCount - 1;
            int i0 = Mathf.FloorToInt(FramePosition);
            int i1 = Mathf.Min(i0 + 1, last);
            float a = FramePosition - i0;
            GhostInstantaneousMetrics m0 = _data.frames[i0].instantaneous_metrics;
            GhostInstantaneousMetrics m1 = _data.frames[i1].instantaneous_metrics;
            m.latent_error = Mathf.Lerp(m0.latent_error, m1.latent_error, a);
            m.position_abs_error = Mathf.Lerp(m0.position_abs_error, m1.position_abs_error, a);
            m.velocity_abs_error = Mathf.Lerp(m0.velocity_abs_error, m1.velocity_abs_error, a);
            m.velocity_relative_degradation = Mathf.Lerp(m0.velocity_relative_degradation, m1.velocity_relative_degradation, a);
            m.position_relative_degradation = Mathf.Lerp(m0.position_relative_degradation, m1.position_relative_degradation, a);

            // Smoothly ground telemetry errors to 0 at frame 1 matching the visual warmup blend
            float warmupBlend = SmoothStepMinimumJerk(FramePosition / 4.0f);
            m.latent_error *= warmupBlend;
            m.position_abs_error *= warmupBlend;
            m.velocity_abs_error *= warmupBlend;
            return m;
        }

        private void HandleHotkeys()
        {
#if ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKeyDown(KeyCode.Space)) TogglePause();
            if (Input.GetKeyDown(KeyCode.R)) Restart();
            if (Input.GetKeyDown(KeyCode.E)) ToggleTargetElevationPlane();
            if (Input.GetKeyDown(KeyCode.T))
            {
                randomizeTargetElevation = true;
                SetTargetElevationPlane((TargetElevationPlane)Random.Range(0, 3), disableRandomization: false);
            }
#endif
        }
    }
}
