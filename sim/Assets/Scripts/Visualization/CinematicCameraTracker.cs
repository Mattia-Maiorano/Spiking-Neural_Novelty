using UnityEngine;

namespace SPWM.Visualization
{
    /// <summary>
    /// Damped isometric follow camera for ghost-execution recordings.
    /// Tracks the ground-truth rig's centre of mass with <see cref="Vector3.SmoothDamp"/>, keeps a fixed
    /// elevated/angled viewpoint (default 30 deg elevation, 45 deg azimuth) and auto-adjusts distance so
    /// the union of Target + Ghost bounds always stays inside the frustum. A screen-composition offset
    /// pushes the robot away from the HUD's negative space.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    [DisallowMultipleComponent]
    public sealed class CinematicCameraTracker : MonoBehaviour
    {
        [Header("Target")]
        [SerializeField] private GhostPlaybackManager playback;
        [Tooltip("Fallback target if no GhostPlaybackManager is assigned.")]
        [SerializeField] private Transform fallbackTarget;

        [Header("Isometric Framing")]
        [SerializeField, Range(0f, 89f)] private float elevationDeg = 30f;
        [SerializeField, Range(-180f, 180f)] private float azimuthDeg = 45f;
        [SerializeField, Min(0.5f)] private float minDistance = 3.0f;
        [SerializeField, Min(0.5f)] private float maxDistance = 9.0f;
        [Tooltip("Extra margin around the robots' bounding sphere (1 = tight).")]
        [SerializeField, Range(1f, 2.5f)] private float framingPadding = 1.35f;
        [Tooltip("Where the robot sits on screen, in viewport units from centre (x: -0.5..0.5, y: -0.5..0.5). " +
                 "Positive X moves the robot right, leaving the left third free for the HUD.")]
        [SerializeField] private Vector2 screenComposition = new Vector2(0.12f, 0.05f);
        [SerializeField] private Vector3 focusOffset = new Vector3(0f, 0.1f, 0f);

        [Header("Damping")]
        [SerializeField, Min(0.01f)] private float positionSmoothTime = 0.45f;
        [SerializeField, Min(0.01f)] private float focusSmoothTime = 0.30f;
        [SerializeField, Min(0.01f)] private float distanceSmoothTime = 0.8f;

        private Camera _cam;
        private Vector3 _focus, _focusVel;
        private Vector3 _posVel;
        private float _distance, _distanceVel;
        private bool _initialised;

        private void Awake()
        {
            _cam = GetComponent<Camera>();
        }

        private void OnEnable()
        {
            if (playback != null) playback.PlaybackRestarted += OnPlaybackRestarted;
        }

        private void OnDisable()
        {
            if (playback != null) playback.PlaybackRestarted -= OnPlaybackRestarted;
        }

        private void OnPlaybackRestarted(GhostPlaybackManager _) => _initialised = false; // hard cut on loop

        private void LateUpdate()
        {
            if (!TryGetFraming(out Vector3 com, out float radius)) return;

            Quaternion orbit = Quaternion.Euler(elevationDeg, azimuthDeg, 0f);
            float desiredDistance = Mathf.Clamp(RequiredDistance(radius), minDistance, maxDistance);
            Vector3 desiredFocus = com + focusOffset;

            if (!_initialised)
            {
                _focus = desiredFocus;
                _distance = desiredDistance;
                _focusVel = _posVel = Vector3.zero;
                _distanceVel = 0f;
                transform.SetPositionAndRotation(desiredFocus - orbit * Vector3.forward * desiredDistance, orbit);
                _initialised = true;
            }

            _focus = Vector3.SmoothDamp(_focus, desiredFocus, ref _focusVel, focusSmoothTime);
            _distance = Mathf.SmoothDamp(_distance, desiredDistance, ref _distanceVel, distanceSmoothTime);

            // Composition: shift the camera's aim so the focus lands at `screenComposition` in the viewport.
            float frustumH = 2f * _distance * Mathf.Tan(0.5f * _cam.fieldOfView * Mathf.Deg2Rad);
            float frustumW = frustumH * _cam.aspect;
            Vector3 right = orbit * Vector3.right, up = orbit * Vector3.up;
            Vector3 aim = _focus - right * (screenComposition.x * frustumW) - up * (screenComposition.y * frustumH);

            Vector3 desiredPos = aim - orbit * Vector3.forward * _distance;
            transform.position = Vector3.SmoothDamp(transform.position, desiredPos, ref _posVel, positionSmoothTime);
            transform.rotation = orbit; // fixed isometric orientation: no rotational jitter
        }

        private bool TryGetFraming(out Vector3 com, out float radius)
        {
            if (playback != null && playback.IsLoaded && playback.TargetRig != null)
            {
                com = playback.TargetRig.CenterOfMass();
                Bounds b = playback.TargetRig.WorldBounds();
                b.Encapsulate(playback.GhostRig.WorldBounds()); // keep ghost limbs in frame too
                // Radius measured from the COM (not bounds centre) since that is what we track.
                radius = Vector3.Distance(com, b.center) + b.extents.magnitude;
                return true;
            }
            if (fallbackTarget != null)
            {
                com = fallbackTarget.position;
                radius = 1.2f;
                return true;
            }
            com = Vector3.zero;
            radius = 0f;
            return false;
        }

        /// <summary>Distance so a sphere of <paramref name="radius"/> fits the tighter of vertical/horizontal FOV,
        /// accounting for the composition offset eating into one side of the frame.</summary>
        private float RequiredDistance(float radius)
        {
            float vHalf = 0.5f * _cam.fieldOfView * Mathf.Deg2Rad;
            float hHalf = Mathf.Atan(Mathf.Tan(vHalf) * _cam.aspect);
            float usableV = 1f - 2f * Mathf.Abs(screenComposition.y);
            float usableH = 1f - 2f * Mathf.Abs(screenComposition.x);
            float halfAngle = Mathf.Min(
                Mathf.Atan(Mathf.Tan(vHalf) * Mathf.Max(0.2f, usableV)),
                Mathf.Atan(Mathf.Tan(hHalf) * Mathf.Max(0.2f, usableH)));
            return framingPadding * radius / Mathf.Sin(halfAngle);
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            if (maxDistance < minDistance) maxDistance = minDistance;
            screenComposition.x = Mathf.Clamp(screenComposition.x, -0.4f, 0.4f);
            screenComposition.y = Mathf.Clamp(screenComposition.y, -0.4f, 0.4f);
        }
#endif
    }
}
