using System;
using System.Collections.Generic;
using System.IO;
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
        public float[] grayscale_frame;   // Flattened [H * W] intensity frame
        public float[] dense_events;      // Flattened [2 * H * W] ON/OFF binary spikes
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
        public int sensor_height;
        public int sensor_width;
        public float dt;
        public float event_threshold;
        public int[] phase_flags;
        public List<TimestepTelemetryRecord> steps = new List<TimestepTelemetryRecord>();
    }

    /// <summary>
    /// Ground-Truth Telemetry & Neuromorphic Sensor Exporter for SPWM-v0.5.0 (6-DOF Active Perception Grasping).
    /// Orchestrates 6-DOF manipulator trajectory rollouts in Unity, recording per-timestep:
    ///   - trajectory_id
    ///   - phase_flag (0 = Blind Approach, 1 = Sensory Heft, 2 = Blind Transport)
    ///   - joint_positions (q* in R^6)
    ///   - joint_velocities (p* in R^6)
    ///   - applied motor actions (u_t in R^6)
    ///   - neuromorphic dense event frames / grayscale image sequences
    /// Serializes trajectories to JSON files consumable by spwm.data.datasets.EventWorldDataset.
    /// </summary>
    public class RoboticTelemetryLogger : MonoBehaviour
    {
        [Header("Simulation & Export Settings")]
        [SerializeField] private ArticulatedRobotRig robotRig;
        [SerializeField] private ActionConditionedController controller;
        [SerializeField] private ProceduralObjectSpawner objectSpawner;
        [SerializeField] private NeuromorphicEventBridge eventBridge;
        [SerializeField] private string exportDirectory = "ExportedTrajectories";
        [SerializeField] private int totalTrajectoriesToExport = 100;
        [SerializeField] private int sequenceLength = 105;
        [SerializeField] private float physicsDt = 0.05f;

        private void Awake()
        {
            Time.fixedDeltaTime = physicsDt;
            if (objectSpawner == null && robotRig != null)
            {
                objectSpawner = robotRig.GetComponent<ProceduralObjectSpawner>();
            }
        }

        /// <summary>
        /// Runs and serializes a single deterministic 6-DOF active perception grasping trajectory
        /// for the given trajectoryId.
        /// </summary>
        public ExportedUnityTrajectory RecordAndExportTrajectory(int trajectoryId)
        {
            robotRig.ResetToDeterministicPose(seed: trajectoryId);
            SpawnedPayloadSpec? payloadSpec = null;
            if (objectSpawner != null)
            {
                payloadSpec = objectSpawner.SpawnEpisodePayload(episodeSeed: trajectoryId, rig: robotRig);
            }
            controller.InitializeControllerState(seed: trajectoryId, payloadSpec: payloadSpec);
            eventBridge.ResetSensorBaseline();

            var trajectory = new ExportedUnityTrajectory
            {
                trajectory_id = trajectoryId,
                archetype = robotRig.Archetype.ToString(),
                payload_shape = payloadSpec.HasValue ? payloadSpec.Value.Shape.ToString() : "Box",
                payload_mass = payloadSpec.HasValue ? payloadSpec.Value.Mass : 1.5f,
                sequence_length = sequenceLength,
                joint_count = robotRig.JointCount,
                action_dim = controller.ActionDim,
                sensor_height = eventBridge.SensorHeight,
                sensor_width = eventBridge.SensorWidth,
                dt = physicsDt,
                event_threshold = eventBridge.EventThreshold,
                phase_flags = new int[sequenceLength]
            };

            Physics.autoSimulation = false;
            for (int t = 0; t < sequenceLength; t++)
            {
                float currentTime = t * physicsDt;
                float[] u_t = controller.StepWaypointController(t, sequenceLength, physicsDt);
                int phaseFlag = controller.CurrentPhaseFlag;
                trajectory.phase_flags[t] = phaseFlag;
                controller.ApplyActionsToJoints();

                float[] q_star = robotRig.GetJointPositions();
                float[] p_star = robotRig.GetJointVelocities();
                var (denseEvents, grayscale) = eventBridge.CaptureStep();

                trajectory.steps.Add(new TimestepTelemetryRecord
                {
                    step = t,
                    timestamp = currentTime,
                    phase_flag = phaseFlag,
                    joint_positions = q_star,
                    joint_velocities = p_star,
                    actions = u_t,
                    grayscale_frame = grayscale,
                    dense_events = denseEvents
                });

                Physics.Simulate(physicsDt);
            }
            Physics.autoSimulation = true;

            WriteTrajectoryToDisk(trajectory);
            return trajectory;
        }

        public string WriteTrajectoryToDisk(ExportedUnityTrajectory trajectory)
        {
            string fullDir = Path.IsPathRooted(exportDirectory)
                ? exportDirectory
                : Path.Combine(Application.dataPath, "..", exportDirectory);
            Directory.CreateDirectory(fullDir);

            string filePath = Path.Combine(fullDir, $"trajectory_{trajectory.trajectory_id:D5}.json");
            string json = JsonUtility.ToJson(trajectory, prettyPrint: false);
            File.WriteAllText(filePath, json);
            return filePath;
        }
    }
}
