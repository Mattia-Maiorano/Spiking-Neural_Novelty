using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using UnityEngine;

namespace SPWM.Robotics
{
    [Serializable]
    public class TimestepTelemetryRecord
    {
        public int step;
        public float timestamp;
        public int phase_flag;            // 0 = Blind Approach, 1 = Sensory Heft, 2 = Blind Transport
        public float[] joint_positions;   // q* in R^6
        public float[] joint_velocities;  // p* in R^6
        public float[] actions;           // u_t in R^6
        public float[] grayscale_frame;       // Flattened [H * W] static global intensity frame
        public float[] wrist_grayscale_frame; // Flattened [H * W] egocentric wrist-mounted intensity frame
        public float[] dense_events;          // Flattened [4 * H * W] dual-sensor ON/OFF binary spikes
    }

    [Serializable]
    public class ExportedUnityTrajectory
    {
        public int trajectory_id;
        public string archetype;
        public string payload_shape;
        public float payload_mass;
        public int sequence_length;
        public int joint_count;
        public int action_dim;
        public int sensor_channels;
        public int sensor_height;
        public int sensor_width;
        public float dt;
        public float event_threshold;
        public int[] phase_flags;
        public List<TimestepTelemetryRecord> steps = new List<TimestepTelemetryRecord>();
    }

    /// <summary>
    /// Ground-Truth Telemetry & Dual-Camera Neuromorphic Sensor Exporter for SPWM-v0.6.0 (6-DOF Active Perception Grasping).
    /// Orchestrates 6-DOF manipulator trajectory rollouts in Unity, recording per-timestep:
    ///   - trajectory_id
    ///   - phase_flag (0 = Blind Approach, 1 = Sensory Heft, 2 = Blind Transport)
    ///   - joint_positions (q* in R^6)
    ///   - joint_velocities (p* in R^6)
    ///   - applied motor actions (u_t in R^6)
    ///   - dual-sensor neuromorphic dense event frames ([4 * H * W]) and global + wrist grayscale sequences ([H * W] each)
    /// Serializes trajectories to JSON files consumable by spwm.data.datasets.EventWorldDataset.
    /// </summary>
    public class RoboticTelemetryLogger : MonoBehaviour
    {
        [Header("Simulation & Export Settings")]
        [SerializeField] private ArticulatedRobotRig robotRig;
        [SerializeField] private ActionConditionedController controller;
        [SerializeField] private ProceduralObjectSpawner objectSpawner;
        [SerializeField] private NeuromorphicEventBridge eventBridge;
        [SerializeField] private NeuromorphicEventBridge wristEventBridge;
        [SerializeField] private string exportDirectory = "ExportedTrajectories";
        [SerializeField] private int totalTrajectoriesToExport = 4000;
        [SerializeField] private int sequenceLength = 105;
        [SerializeField] private float physicsDt = 0.05f;
        [Tooltip("If true, compress JSON exports using GZip (.json.gz). Reduces file sizes ~22x.")]
        [SerializeField] private bool compressGzip = true;
        [Tooltip("If false, omits dense_events array from telemetry since it is regenerated on-the-fly by SPWM dataset.")]
        [SerializeField] private bool includeDenseEvents = false;

        [SerializeField] private int physicsSubSteps = 5;

        [Header("Stability Guard")]
        [Tooltip("A step-to-step joint change above this (rad) that also reverses direction counts as an oscillation flip.")]
        [SerializeField] private float flipDeltaThresholdRad = 0.15f;
        [Tooltip("Max allowed oscillation flips per trajectory before it is rejected.")]
        [SerializeField] private int maxAllowedFlips = 3;
        [Tooltip("Max allowed |dq| of any joint in one step (rad).")]
        [SerializeField] private float maxJointStepRad = 0.6f;
        [SerializeField] private int maxRetries = 5;

        public int TotalTrajectoriesToExport
        {
            get => totalTrajectoriesToExport;
            set => totalTrajectoriesToExport = value;
        }

        public string ExportDirectory
        {
            get => exportDirectory;
            set => exportDirectory = value;
        }

        public bool CompressGzip
        {
            get => compressGzip;
            set => compressGzip = value;
        }

        public bool IncludeDenseEvents
        {
            get => includeDenseEvents;
            set => includeDenseEvents = value;
        }

        private void Awake()
        {
            Time.fixedDeltaTime = physicsDt / Mathf.Max(1, physicsSubSteps);
            if (objectSpawner == null && robotRig != null)
            {
                objectSpawner = robotRig.GetComponent<ProceduralObjectSpawner>();
            }
            EnsureWristSensorAttached();
        }

        /// <summary>
        /// Ensures an egocentric wrist-mounted NeuromorphicEventBridge is attached directly to the robot's
        /// end-effector tip and linked to the primary global eventBridge for simultaneous dual-camera capture.
        /// </summary>
        public NeuromorphicEventBridge EnsureWristSensorAttached()
        {
            if (wristEventBridge == null && eventBridge != null && eventBridge.SecondaryWristBridge != null)
            {
                wristEventBridge = eventBridge.SecondaryWristBridge;
            }

            if (wristEventBridge == null && robotRig != null)
            {
                Transform mount = robotRig.WristCameraMountTransform;
                if (mount != null)
                {
                    wristEventBridge = mount.GetComponentInChildren<NeuromorphicEventBridge>();
                    if (wristEventBridge == null)
                    {
                        GameObject wristCamGo = new GameObject("WristNeuromorphicCamera");
                        wristCamGo.transform.SetParent(mount, false);
                        wristCamGo.transform.localPosition = Vector3.zero;
                        wristCamGo.transform.localRotation = Quaternion.identity;

                        Camera wristCam = wristCamGo.AddComponent<Camera>();
                        wristCam.fieldOfView = 65f;
                        wristCam.nearClipPlane = 0.02f;
                        wristCam.farClipPlane = 25f;
                        wristCam.clearFlags = CameraClearFlags.SolidColor;
                        wristCam.backgroundColor = new Color(0.08f, 0.10f, 0.14f, 1f);

                        wristEventBridge = wristCamGo.AddComponent<NeuromorphicEventBridge>();
                        int w = eventBridge != null ? eventBridge.SensorWidth : 32;
                        int h = eventBridge != null ? eventBridge.SensorHeight : 32;
                        wristEventBridge.InitializeBuffers(w, h);
                    }
                }
            }

            if (eventBridge != null && wristEventBridge != null)
            {
                eventBridge.SecondaryWristBridge = wristEventBridge;
            }
            return wristEventBridge;
        }

        /// <summary>
        /// Runs and serializes a single deterministic 6-DOF active perception grasping trajectory
        /// for the given trajectoryId.
        /// </summary>
        public ExportedUnityTrajectory RecordAndExportTrajectory(int trajectoryId)
        {
            ExportedUnityTrajectory trajectory = null;
            for (int attempt = 0; attempt <= maxRetries; attempt++)
            {
                // Attempt 0 uses the canonical seed; retries perturb the seed (episode content/timing) only.
                int seed = trajectoryId + attempt * 100003;
                trajectory = RunRollout(trajectoryId, seed);
                if (IsTrajectoryStable(trajectory, out string reason))
                {
                    break;
                }
                Debug.LogWarning($"[RoboticTelemetryLogger] Trajectory {trajectoryId} (seed {seed}) rejected: {reason}. " +
                                 (attempt < maxRetries ? "Retrying with a new seed." : "Retries exhausted; exporting last attempt."));
            }

            WriteTrajectoryToDisk(trajectory);
            return trajectory;
        }

        /// <summary>
        /// Detects unstable rollouts: NaN/Inf, excessive per-step joint jumps, or repeated sign-alternating
        /// large steps (Nyquist-frequency oscillation).
        /// </summary>
        public bool IsTrajectoryStable(ExportedUnityTrajectory traj, out string reason)
        {
            reason = null;
            int flips = 0;
            for (int t = 1; t < traj.steps.Count; t++)
            {
                float[] q = traj.steps[t].joint_positions;
                float[] qp = traj.steps[t - 1].joint_positions;
                float[] qpp = t >= 2 ? traj.steps[t - 2].joint_positions : null;
                for (int j = 0; j < q.Length; j++)
                {
                    float d = q[j] - qp[j];
                    if (float.IsNaN(d) || float.IsInfinity(d))
                    {
                        reason = $"non-finite joint {j} at step {t}";
                        return false;
                    }
                    if (Mathf.Abs(d) > maxJointStepRad)
                    {
                        reason = $"joint {j} jumped {d:F2} rad at step {t}";
                        return false;
                    }
                    if (qpp != null)
                    {
                        float dPrev = qp[j] - qpp[j];
                        if (d * dPrev < 0f && Mathf.Abs(d) > flipDeltaThresholdRad && ++flips > maxAllowedFlips)
                        {
                            reason = $"oscillation ({flips} flips, last joint {j} step {t})";
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        private ExportedUnityTrajectory RunRollout(int trajectoryId, int seed)
        {
            robotRig.ResetToDeterministicPose(seed: seed);
            SpawnedPayloadSpec? payloadSpec = null;
            if (objectSpawner != null)
            {
                payloadSpec = objectSpawner.SpawnEpisodePayload(episodeSeed: seed, rig: robotRig);
            }
            controller.InitializeControllerState(seed: seed, payloadSpec: payloadSpec);
            EnsureWristSensorAttached();
            eventBridge.ResetSensorBaseline();
            if (wristEventBridge != null && wristEventBridge != eventBridge.SecondaryWristBridge)
            {
                wristEventBridge.ResetSensorBaseline();
            }

            var trajectory = new ExportedUnityTrajectory
            {
                trajectory_id = trajectoryId,
                archetype = robotRig.Archetype.ToString(),
                payload_shape = payloadSpec.HasValue ? payloadSpec.Value.Shape.ToString() : "Box",
                payload_mass = payloadSpec.HasValue ? payloadSpec.Value.Mass : 1.5f,
                sequence_length = sequenceLength,
                joint_count = robotRig.JointCount,
                action_dim = controller.ActionDim,
                sensor_channels = 4,
                sensor_height = eventBridge.SensorHeight,
                sensor_width = eventBridge.SensorWidth,
                dt = physicsDt,
                event_threshold = eventBridge.EventThreshold,
                phase_flags = new int[sequenceLength]
            };

            Physics.simulationMode = SimulationMode.Script;
            int subSteps = Mathf.Max(1, physicsSubSteps);
            float subStepDt = physicsDt / subSteps;

            for (int t = 0; t < sequenceLength; t++)
            {
                float currentTime = t * physicsDt;
                float[] u_t = controller.StepWaypointController(t, sequenceLength, physicsDt);
                int phaseFlag = controller.CurrentPhaseFlag;
                trajectory.phase_flags[t] = phaseFlag;
                controller.ApplyActionsToJoints();

                float[] q_star = robotRig.GetJointPositions();
                float[] p_star = robotRig.GetJointVelocities();
                var (unifiedDenseEvents, globalGrayscale, wristGrayscale, _) = eventBridge.CaptureDualStep(wristEventBridge);

                trajectory.steps.Add(new TimestepTelemetryRecord
                {
                    step = t,
                    timestamp = currentTime,
                    phase_flag = phaseFlag,
                    joint_positions = q_star,
                    joint_velocities = p_star,
                    actions = u_t,
                    grayscale_frame = globalGrayscale,
                    wrist_grayscale_frame = wristGrayscale,
                    dense_events = includeDenseEvents ? unifiedDenseEvents : null
                });

                for (int s = 0; s < subSteps; s++)
                {
                    Physics.Simulate(subStepDt);
                }
            }
            Physics.simulationMode = SimulationMode.FixedUpdate;

            return trajectory;
        }

        public string WriteTrajectoryToDisk(ExportedUnityTrajectory trajectory)
        {
            string fullDir = Path.IsPathRooted(exportDirectory)
                ? exportDirectory
                : Path.Combine(Application.dataPath, "..", exportDirectory);
            Directory.CreateDirectory(fullDir);

            string json = JsonUtility.ToJson(trajectory, prettyPrint: false);

            if (compressGzip)
            {
                string filePath = Path.Combine(fullDir, $"trajectory_{trajectory.trajectory_id:D5}.json.gz");
                byte[] rawBytes = Encoding.UTF8.GetBytes(json);
                using (var fileStream = File.Create(filePath))
                using (var gzipStream = new GZipStream(fileStream, CompressionMode.Compress))
                {
                    gzipStream.Write(rawBytes, 0, rawBytes.Length);
                }
                return filePath;
            }
            else
            {
                string filePath = Path.Combine(fullDir, $"trajectory_{trajectory.trajectory_id:D5}.json");
                File.WriteAllText(filePath, json);
                return filePath;
            }
        }

        /// <summary>
        /// Sequentially runs and exports all totalTrajectoriesToExport trajectories
        /// with procedural temporal and distance randomization.
        /// </summary>
        [ContextMenu("Export Full Dataset Batch")]
        public void ExportBatchTrajectories()
        {
            Debug.Log($"[RoboticTelemetryLogger] Starting batch export of {totalTrajectoriesToExport} trajectories into '{exportDirectory}'...");
            float startRealTime = Time.realtimeSinceStartup;
            for (int id = 0; id < totalTrajectoriesToExport; id++)
            {
                RecordAndExportTrajectory(id);
                if ((id + 1) % 50 == 0 || id == totalTrajectoriesToExport - 1)
                {
                    Debug.Log($"[RoboticTelemetryLogger] Exported {id + 1}/{totalTrajectoriesToExport} trajectories.");
                }
            }
            float elapsed = Time.realtimeSinceStartup - startRealTime;
            Debug.Log($"[RoboticTelemetryLogger] Successfully exported {totalTrajectoriesToExport} trajectories in {elapsed:F1}s.");
        }
    }
}
