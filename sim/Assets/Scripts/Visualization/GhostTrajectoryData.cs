using System;
using System.IO;
using UnityEngine;

namespace SPWM.Visualization
{
    /// <summary>
    /// C# mirror of the Layer 3 ghost-execution contract written by
    /// <c>spwm.evaluation.visualizer.GhostVisualizerExporter</c> (schema <c>spwm.ghost_trajectory/1.0</c>).
    /// Field names intentionally use snake_case so <see cref="JsonUtility"/> binds them 1:1.
    /// </summary>
    [Serializable]
    public sealed class GhostInstantaneousMetrics
    {
        public float position_abs_error;            // ||q_hat_t - q*_t||_2 (rad)
        public float velocity_abs_error;            // ||p_hat_t - p*_t||_2 (rad/s)
        public float velocity_relative_degradation; // E_v(t) / E_v(1)
        public float position_relative_degradation; // E_q(t) / E_q(1)
    }

    [Serializable]
    public sealed class GhostFrame
    {
        public int t;
        public float time_s;
        public float[] ground_truth_q;
        public float[] predicted_q;
        public float[] ground_truth_p;
        public float[] predicted_p;
        public GhostInstantaneousMetrics instantaneous_metrics;
    }

    [Serializable]
    public sealed class GhostSummary
    {
        public float mean_position_abs_error;
        public float final_position_abs_error;
        public float max_position_abs_error;
        public float mean_velocity_abs_error;
        public float final_velocity_abs_error;
        public float max_velocity_abs_error;
        public float final_velocity_relative_degradation;
        public float final_position_relative_degradation;
    }

    [Serializable]
    public sealed class GhostMetadata
    {
        public string schema;
        public string model_version;
        public int horizon;
        public float dt;
        public int joint_count;
        public string[] joint_names;
        public string angle_unit;
        public bool has_velocities;
        public int t_start;
        public int sample_index;
        public int trajectory_id;
        public int batch_index;
        public int grasp_lock_frame_index = -1;
        public string exported_at;
        public GhostSummary summary;
    }

    [Serializable]
    public sealed class GhostTrajectoryData
    {
        public const string ExpectedSchemaPrefix = "spwm.ghost_trajectory/1.";

        public GhostMetadata metadata;
        public GhostFrame[] frames;

        public int FrameCount => frames != null ? frames.Length : 0;
        public float Dt => metadata != null && metadata.dt > 0f ? metadata.dt : 0.05f;
        public int JointCount => metadata != null ? metadata.joint_count : 0;

        /// <summary>Loads and validates a ghost trajectory from <c>Application.streamingAssetsPath</c>.</summary>
        /// <remarks>Uses synchronous file IO: valid for Editor / Standalone (the recording targets).
        /// Android/WebGL would require UnityWebRequest because StreamingAssets is packed there.</remarks>
        public static GhostTrajectoryData LoadFromStreamingAssets(string fileName)
        {
            string path = Path.Combine(Application.streamingAssetsPath, fileName);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Ghost trajectory not found at '{path}'. Run: python experiments/export_ghost_trajectory.py", path);
            }
            return Parse(File.ReadAllText(path));
        }

        public static GhostTrajectoryData Parse(string json)
        {
            GhostTrajectoryData data = JsonUtility.FromJson<GhostTrajectoryData>(json);
            data.Validate();
            return data;
        }

        public static GhostTrajectoryData FromJson(string json)
        {
            return Parse(json);
        }

        public void Validate()
        {
            if (metadata == null || frames == null || frames.Length == 0)
                throw new FormatException("Ghost trajectory is missing 'metadata' or 'frames'.");
            if (string.IsNullOrEmpty(metadata.schema) || !metadata.schema.StartsWith(ExpectedSchemaPrefix, StringComparison.Ordinal))
                Debug.LogWarning($"[Ghost] Unexpected schema '{metadata.schema}', expected '{ExpectedSchemaPrefix}x'.");
            if (frames.Length != metadata.horizon)
                Debug.LogWarning($"[Ghost] frames.Length={frames.Length} != metadata.horizon={metadata.horizon}.");

            for (int i = 0; i < frames.Length; i++)
            {
                GhostFrame f = frames[i];
                if (f.ground_truth_q == null || f.predicted_q == null ||
                    f.ground_truth_q.Length != metadata.joint_count || f.predicted_q.Length != metadata.joint_count)
                {
                    throw new FormatException($"Frame {i}: q arrays must have joint_count={metadata.joint_count} entries.");
                }
                if (f.instantaneous_metrics == null) f.instantaneous_metrics = new GhostInstantaneousMetrics();
            }
        }
    }
}
