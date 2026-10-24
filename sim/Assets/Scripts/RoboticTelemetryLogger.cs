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
        public float[] joint_positions;   // q* in R^{joint_count}
        public float[] joint_velocities;  // p* in R^{joint_count}
        public float[] actions;           // u_t in R^{action_dim}
        public float[] grayscale_frame;   // Flattened [H * W] intensity frame
        public float[] dense_events;      // Flattened [2 * H * W] ON/OFF binary spikes
    }

    [Serializable]
    public class ExportedUnityTrajectory
    {
        public int trajectory_id;
        public string archetype;
        public string attractor_regime;
        public int sequence_length;
        public int joint_count;
        public int action_dim;
        public int sensor_height;
        public int sensor_width;
        public float dt;
        public float event_threshold;
        public List<TimestepTelemetryRecord> steps = new List<TimestepTelemetryRecord>();
    }

    /// <summary>
    /// Ground-Truth Telemetry & Neuromorphic Sensor Exporter for SPWM-v0.4.0.
    /// Orchestrates trajectory rollouts in Unity, recording per-timestep:
    ///   - trajectory_id
    ///   - joint_positions (q*)
    ///   - joint_velocities (p*)
    ///   - applied motor actions (u_t)
    ///   - neuromorphic dense event frames / grayscale image sequences
    /// Serializes trajectories to JSON files consumable by spwm.data.datasets.EventWorldDataset.
    /// </summary>
    public class RoboticTelemetryLogger : MonoBehaviour
    {
        [Header("Simulation & Export Settings")]
        [SerializeField] private ArticulatedRobotRig robotRig;
        [SerializeField] private ActionConditionedController controller;
        [SerializeField] private NeuromorphicEventBridge eventBridge;
        [SerializeField] private string exportDirectory = "ExportedTrajectories";
        [SerializeField] private int totalTrajectoriesToExport = 100;
        [SerializeField] private int sequenceLength = 105;
        [SerializeField] private float physicsDt = 0.05f;

        private void Awake()
        {
            Time.fixedDeltaTime = physicsDt;
        }

        /// <summary>
        /// Runs and serializes a single deterministic trajectory for the given trajectoryId.
        /// Alternates between limit-cycle walking gaits (even IDs) and fixed-point posture balancing (odd IDs).
        /// </summary>
        public ExportedUnityTrajectory RecordAndExportTrajectory(int trajectoryId)
        {
            robotRig.ResetToDeterministicPose(seed: trajectoryId);
            controller.InitializeControllerState(seed: trajectoryId);
            controller.CurrentRegime = (trajectoryId % 2 == 0)
                ? AttractorRegime.LimitCycleWalkingGait
                : AttractorRegime.FixedPointBalancing;
            eventBridge.ResetSensorBaseline();

            var trajectory = new ExportedUnityTrajectory
            {
                trajectory_id = trajectoryId,
                archetype = robotRig.Archetype.ToString(),
                attractor_regime = controller.CurrentRegime.ToString(),
                sequence_length = sequenceLength,
                joint_count = robotRig.JointCount,
                action_dim = controller.ActionDim,
                sensor_height = eventBridge.SensorHeight,
                sensor_width = eventBridge.SensorWidth,
                dt = physicsDt,
                event_threshold = eventBridge.EventThreshold
            };

            Physics.autoSimulation = false;
            for (int t = 0; t < sequenceLength; t++)
            {
                float currentTime = t * physicsDt;
                float[] u_t = controller.SampleRegimeAction(currentTime, trajectoryId);
                controller.ApplyActionsToJoints();

                float[] q_star = robotRig.GetJointPositions();
                float[] p_star = robotRig.GetJointVelocities();
                var (denseEvents, grayscale) = eventBridge.CaptureStep();

                trajectory.steps.Add(new TimestepTelemetryRecord
                {
                    step = t,
                    timestamp = currentTime,
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
