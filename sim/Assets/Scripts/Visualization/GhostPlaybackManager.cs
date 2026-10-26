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

        private void CreateVisual(Transform pivot, ArticulatedLinkSpec spec, Material material, bool isRoot, bool isEndEffector)
        {
            Vector3 d = spec.LinkDimensions;

            if (isRoot)
            {
                // Heavy bolted floor flange + pedestal column spanning [0, d.y]
                GameObject flange = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                flange.name = spec.Name + "_FloorFlange";
                Object.Destroy(flange.GetComponent<Collider>());
                flange.transform.SetParent(pivot, false);
                flange.transform.localScale = new Vector3(d.x * 1.45f, 0.015f, d.z * 1.45f);
                flange.transform.localPosition = new Vector3(0f, 0.015f, 0f);
                RegisterRenderer(flange.GetComponent<Renderer>(), material);
            }
            else
            {
                // Spherical bolted joint housing centered at the exact pivot (0, 0, 0) so links never visually gap
                GameObject hub = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                hub.name = spec.Name + "_JointHub";
                Object.Destroy(hub.GetComponent<Collider>());
                hub.transform.SetParent(pivot, false);
                float hubDiameter = Mathf.Max(d.x, d.z) * 1.22f;
                hub.transform.localScale = new Vector3(hubDiameter, hubDiameter, hubDiameter);
                hub.transform.localPosition = Vector3.zero;
                RegisterRenderer(hub.GetComponent<Renderer>(), material);
            }

            // Rigid cylindrical structural shaft spanning strictly from the joint pivot (0, 0, 0) to child pivot (0, d.y, 0).
            // Unity's Cylinder primitive has height = 2.0, so localScale.y = 0.5 * d.y and localPosition.y = 0.5 * d.y.
            GameObject shaft = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            shaft.name = spec.Name + "_VisualShaft";
            Object.Destroy(shaft.GetComponent<Collider>());
            shaft.transform.SetParent(pivot, false);
            shaft.transform.localScale = new Vector3(d.x, 0.5f * d.y, d.z);
            shaft.transform.localPosition = new Vector3(0f, 0.5f * d.y, 0f);
            shaft.transform.localRotation = Quaternion.identity;
            RegisterRenderer(shaft.GetComponent<Renderer>(), material);

            if (isEndEffector)
            {
                // Magnetic gripper tip anchored at the distal end of Wrist_Roll (0, d.y, 0)
                var tip = new GameObject("EndEffectorTip").transform;
                tip.SetParent(pivot, false);
                tip.localPosition = new Vector3(0f, d.y, 0f);
                tip.localRotation = Quaternion.identity;
                tip.localScale = Vector3.one;
                EndEffectorTip = tip;

                GameObject magPlate = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                magPlate.name = "MagneticGripperPlate";
                Object.Destroy(magPlate.GetComponent<Collider>());
                magPlate.transform.SetParent(tip, false);
                magPlate.transform.localScale = new Vector3(d.x * 1.45f, 0.008f, d.z * 1.45f);
                magPlate.transform.localPosition = Vector3.zero;
                RegisterRenderer(magPlate.GetComponent<Renderer>(), material);
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
        [SerializeField] private string trajectoryFileName = "trajectory_ghost_v0_4_0.json";

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

        [Header("Grasp & Payload Visualization")]
        [Tooltip("Playback frame index (0-indexed) at which the end-effector locks onto the ground object (start of Heft).")]
        [SerializeField] private int graspLockFrameIndex = 35;
        [SerializeField] private Vector3 payloadDimensions = new Vector3(0.11f, 0.07f, 0.09f);

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
        private Transform _payloadVisual;
        private Vector3 _payloadGroundPosition;
        private Quaternion _payloadGroundRotation;
        private Vector3 _payloadGraspLocalPos;
        private Quaternion _payloadGraspLocalRot;
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

        /// <summary>Continuous frame index in [0, FrameCount-1]; fractional part = interpolation alpha.</summary>
        public float FramePosition { get; private set; }

        /// <summary>1-indexed horizon step currently displayed (t in "t / H").</summary>
        public int CurrentStep => _data == null ? 0 : Mathf.Clamp(Mathf.FloorToInt(FramePosition) + 1, 1, FrameCount);

        /// <summary>Normalised progress through the horizon in [0, 1].</summary>
        public float Progress => FrameCount > 1 ? FramePosition / (FrameCount - 1) : 0f;

        public event System.Action<GhostPlaybackManager> Loaded;
        public event System.Action<GhostPlaybackManager> PlaybackRestarted;

        private void Awake()
        {
            try
            {
                _data = GhostTrajectoryData.LoadFromStreamingAssets(trajectoryFileName);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[GhostPlaybackManager] {e.Message}", this);
                enabled = false;
                return;
            }

            EnsureMaterials();
            _target = new KinematicRobotRig("TargetRig_GroundTruth", transform, archetype, targetMaterial, Vector3.zero, forceSagittalAxis);
            _ghost = new KinematicRobotRig("GhostRig_SPWM", transform, archetype, ghostMaterial, ghostOffset, forceSagittalAxis);
            _ghost.SetShadowCasting(false); // holograms do not cast shadows

            if (_target.JointCount != _data.JointCount)
            {
                Debug.LogError($"[GhostPlaybackManager] Rig has {_target.JointCount} actuated joints but data has " +
                               $"{_data.JointCount}. Check archetype.", this);
                enabled = false;
                return;
            }

            PrecomputeRotations();
            InitializeEnvironmentAndPayload();
            ApplyPose(0f);
            Loaded?.Invoke(this);
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

        private void PrecomputeRotations()
        {
            int frames = _data.FrameCount, joints = _data.JointCount;
            _targetRot = new Quaternion[frames, joints];
            _ghostRot = new Quaternion[frames, joints];
            for (int f = 0; f < frames; f++)
            {
                GhostFrame fr = _data.frames[f];
                for (int j = 0; j < joints; j++)
                {
                    _targetRot[f, j] = _target.JointRotation(j, fr.ground_truth_q[j], angleScale);
                    _ghostRot[f, j] = _ghost.JointRotation(j, fr.predicted_q[j], angleScale);
                }
            }
        }

        /// <summary>
        /// Instantiates the industrial ground plane and the target payload sitting on the ground at the exact
        /// 3D location reached by the arm at the grasp frame, so the arm approaches it, locks onto it,
        /// performs a sharp upward Heft lift, and carries + twists it upright to the final target location.
        /// </summary>
        private void InitializeEnvironmentAndPayload()
        {
            // 1. Industrial ground plate at y = 0 where the robot base is bolted
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Cube);
            ground.name = "IndustrialWorkcellFloor";
            Object.Destroy(ground.GetComponent<Collider>());
            ground.transform.SetParent(transform, false);
            ground.transform.localScale = new Vector3(1.6f, 0.016f, 1.6f);
            ground.transform.localPosition = new Vector3(0f, -0.008f, 0f);
            var groundRenderer = ground.GetComponent<Renderer>();
            if (groundRenderer != null)
            {
                Shader s = Shader.Find("Standard");
                if (s != null)
                {
                    var floorMat = new Material(s)
                    {
                        name = "WorkcellFloorMat (runtime)",
                        color = new Color(0.11f, 0.13f, 0.16f)
                    };
                    floorMat.SetFloat("_Glossiness", 0.18f);
                    floorMat.SetFloat("_Metallic", 0.1f);
                    groundRenderer.sharedMaterial = floorMat;
                }
            }

            // 2. Pose the target rig at graspLockFrameIndex to read the exact world-space gripper tip pose on contact
            int lockFrame = Mathf.Clamp(graspLockFrameIndex, 0, _data.FrameCount - 1);
            for (int j = 0; j < _data.JointCount; j++)
            {
                _target.SetJointLocalRotation(j, _targetRot[lockFrame, j]);
            }

            Transform tip = _target.EndEffectorTip;
            Vector3 tipWorldPos = tip != null ? tip.position : (transform.position + new Vector3(0.24f, 0.07f, 0.28f));
            Quaternion tipWorldRot = tip != null ? tip.rotation : Quaternion.identity;

            // Place the payload sitting flat on the ground directly beneath the gripper tip at the grasp moment
            float halfHeight = 0.5f * payloadDimensions.y;
            float groundCenterY = Mathf.Max(halfHeight, tipWorldPos.y - halfHeight);
            _payloadGroundPosition = new Vector3(tipWorldPos.x, groundCenterY, tipWorldPos.z);
            float yawDeg = Mathf.Atan2(tipWorldPos.x - transform.position.x, tipWorldPos.z - transform.position.z) * Mathf.Rad2Deg;
            _payloadGroundRotation = Quaternion.Euler(0f, yawDeg, 0f);

            // Compute exact rigid-lock offset relative to the gripper tip at the moment of magnetic lock
            _payloadGraspLocalPos = Quaternion.Inverse(tipWorldRot) * (_payloadGroundPosition - tipWorldPos);
            _payloadGraspLocalRot = Quaternion.Inverse(tipWorldRot) * _payloadGroundRotation;

            // 3. Instantiate the visual target payload block
            GameObject payloadObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
            payloadObj.name = "TargetPayload_HeavyBlock";
            Object.Destroy(payloadObj.GetComponent<Collider>());
            payloadObj.transform.SetParent(transform, true);
            payloadObj.transform.localScale = payloadDimensions;
            payloadObj.transform.SetPositionAndRotation(_payloadGroundPosition, _payloadGroundRotation);
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
            m.position_abs_error = Mathf.Lerp(m0.position_abs_error, m1.position_abs_error, a);
            m.velocity_abs_error = Mathf.Lerp(m0.velocity_abs_error, m1.velocity_abs_error, a);
            m.velocity_relative_degradation = Mathf.Lerp(m0.velocity_relative_degradation, m1.velocity_relative_degradation, a);
            m.position_relative_degradation = Mathf.Lerp(m0.position_relative_degradation, m1.position_relative_degradation, a);
            return m;
        }

        private void HandleHotkeys()
        {
#if ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKeyDown(KeyCode.Space)) TogglePause();
            if (Input.GetKeyDown(KeyCode.R)) Restart();
#endif
        }
    }
}
