using System.Collections.Generic;
using UnityEngine;
using SPWM.Robotics;

namespace SPWM.Visualization
{
    /// <summary>
    /// Purely kinematic (physics-free) replica of <see cref="ArticulatedRobotRig"/> for playback.
    /// Built from the same <see cref="ArticulatedRobotRig.GetArchetypeSpecs"/> table, so joint order
    /// matches the q* columns of the exported telemetry exactly.
    /// Each link = unscaled pivot transform (rotates) + scaled visual child (renders).
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
        public int JointCount => _jointPivots.Count;
        public IReadOnlyList<Renderer> Renderers => _renderers;

        public KinematicRobotRig(string name, Transform parent, RobotArchetype archetype, Material material,
                                 Vector3 localOffset, bool forceSagittalAxis)
        {
            Root = new GameObject(name).transform;
            Root.SetParent(parent, false);
            Root.localPosition = localOffset;

            List<ArticulatedLinkSpec> specs = ArticulatedRobotRig.GetArchetypeSpecs(archetype);
            var created = new List<Transform>(specs.Count);

            foreach (ArticulatedLinkSpec spec in specs)
            {
                var pivot = new GameObject(spec.Name).transform;
                pivot.SetParent(spec.ParentIndex < 0 ? Root : created[spec.ParentIndex], false);
                pivot.localPosition = spec.LocalAnchorPosition;
                pivot.localRotation = Quaternion.identity;

                CreateVisual(pivot, spec, material, isRoot: spec.ParentIndex < 0);
                created.Add(pivot);
                _links.Add(pivot);
                _linkMasses.Add(Mathf.Max(0.01f, spec.Mass));
                _totalMass += Mathf.Max(0.01f, spec.Mass);

                if (spec.ParentIndex >= 0)
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

        private void CreateVisual(Transform pivot, ArticulatedLinkSpec spec, Material material, bool isRoot)
        {
            Vector3 d = spec.LinkDimensions;
            bool elongatedY = d.y >= d.x && d.y >= d.z;
            GameObject vis = GameObject.CreatePrimitive(elongatedY ? PrimitiveType.Capsule : PrimitiveType.Cube);
            vis.name = spec.Name + "_Visual";
            Object.Destroy(vis.GetComponent<Collider>()); // visual only: no physics interaction between rigs

            vis.transform.SetParent(pivot, false);
            vis.transform.localScale = d;
            // Capsule primitive height = 2 * scale.y, cube height = scale.y. Hang links below their joint.
            float halfHeight = elongatedY ? d.y : 0.5f * d.y;
            vis.transform.localPosition = isRoot ? Vector3.zero : new Vector3(0f, -halfHeight, 0f);

            var r = vis.GetComponent<Renderer>();
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
    /// Ghost execution playback: instantiates an opaque Target rig driven by ground_truth_q and a
    /// holographic Ghost rig driven by predicted_q, interpolating joint rotations between the
    /// discrete simulation steps (dt = 50 ms) with <see cref="Quaternion.Slerp"/>.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class GhostPlaybackManager : MonoBehaviour
    {
        [Header("Data")]
        [SerializeField] private string trajectoryFileName = "trajectory_ghost_v0_4_0.json";

        [Header("Rig")]
        [SerializeField] private RobotArchetype archetype = RobotArchetype.Humanoid;
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
                targetMaterial.SetFloat("_Glossiness", 0.12f);
                targetMaterial.SetFloat("_Metallic", 0.0f);
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
