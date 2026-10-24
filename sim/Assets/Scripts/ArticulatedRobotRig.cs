using System;
using System.Collections.Generic;
using UnityEngine;

namespace SPWM.Robotics
{
    /// <summary>
    /// Supported robotic morphology archetypes for the generalized ArticulationBody rig.
    /// </summary>
    public enum RobotArchetype
    {
        Humanoid = 0,
        Quadruped = 1,
        ManipulatorArm = 2
    }

    /// <summary>
    /// Specification for a single articulated link in the kinematic hierarchy.
    /// </summary>
    [Serializable]
    public struct ArticulatedLinkSpec
    {
        public string Name;
        public int ParentIndex;
        public Vector3 LocalAnchorPosition;
        public Vector3 LinkDimensions;
        public float Mass;
        public float JointFriction;
        public float AngularDamping;
        public float LowerLimitDeg;
        public float UpperLimitDeg;
        public float DriveStiffness;
        public float DriveDamping;
        public float MaxForceLimit;
        public Vector3 AxisRotationEuler;
    }

    /// <summary>
    /// Generalized Articulated Robotic Rig built on PhysX 4 Featherstone ArticulationBody chains.
    /// Optimized for bipedal humanoids while adaptable to quadrupeds and fixed-base manipulator arms.
    /// Models realistic kinematic chains, joint coulomb/viscous friction, and link mass distributions.
    /// </summary>
    public class ArticulatedRobotRig : MonoBehaviour
    {
        [Header("Morphology Configuration")]
        [SerializeField] private RobotArchetype archetype = RobotArchetype.Humanoid;
        [SerializeField] private bool immobileBase = false;
        [SerializeField] private float globalJointFriction = 0.08f;
        [SerializeField] private float globalAngularDamping = 0.15f;
        [SerializeField] private float globalLinearDamping = 0.05f;
        [SerializeField] private float groundStaticFriction = 0.85f;
        [SerializeField] private float groundDynamicFriction = 0.75f;

        private ArticulationBody _rootBody;
        private readonly List<ArticulationBody> _actuatedJoints = new List<ArticulationBody>();
        private PhysicMaterial _groundContactMaterial;

        public RobotArchetype Archetype => archetype;
        public ArticulationBody RootBody => _rootBody;
        public IReadOnlyList<ArticulationBody> ActuatedJoints => _actuatedJoints;
        public int JointCount => _actuatedJoints.Count;

        private void Awake()
        {
            BuildContactMaterial();
            if (_actuatedJoints.Count == 0)
            {
                BuildProceduralRig(archetype);
            }
        }

        private void BuildContactMaterial()
        {
            _groundContactMaterial = new PhysicMaterial("RoboticGroundContact")
            {
                staticFriction = groundStaticFriction,
                dynamicFriction = groundDynamicFriction,
                bounciness = 0.0f,
                frictionCombine = PhysicMaterialCombine.Multiply,
                bounceCombine = PhysicMaterialCombine.Minimum
            };
        }

        /// <summary>
        /// Constructs or reconfigures the ArticulationBody hierarchy according to the selected archetype.
        /// </summary>
        public void BuildProceduralRig(RobotArchetype targetArchetype)
        {
            archetype = targetArchetype;
            _actuatedJoints.Clear();

            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                DestroyImmediate(transform.GetChild(i).gameObject);
            }

            List<ArticulatedLinkSpec> specs = GetArchetypeSpecs(targetArchetype);
            var createdBodies = new List<ArticulationBody>(specs.Count);

            for (int i = 0; i < specs.Count; i++)
            {
                ArticulatedLinkSpec spec = specs[i];
                GameObject linkObj = GameObject.CreatePrimitive(PrimitiveType.Capsule);
                linkObj.name = spec.Name;

                if (spec.ParentIndex < 0)
                {
                    linkObj.transform.SetParent(transform, false);
                    linkObj.transform.localPosition = spec.LocalAnchorPosition;
                }
                else
                {
                    Transform parentTransform = createdBodies[spec.ParentIndex].transform;
                    linkObj.transform.SetParent(parentTransform, false);
                    linkObj.transform.localPosition = spec.LocalAnchorPosition;
                }

                linkObj.transform.localScale = spec.LinkDimensions;
                Collider col = linkObj.GetComponent<Collider>();
                if (col != null && _groundContactMaterial != null)
                {
                    col.sharedMaterial = _groundContactMaterial;
                }

                ArticulationBody ab = linkObj.AddComponent<ArticulationBody>();
                ab.mass = spec.Mass;
                ab.linearDamping = globalLinearDamping;
                ab.angularDamping = spec.AngularDamping > 0f ? spec.AngularDamping : globalAngularDamping;
                ab.jointFriction = spec.JointFriction > 0f ? spec.JointFriction : globalJointFriction;
                ab.useGravity = true;

                if (spec.ParentIndex < 0)
                {
                    ab.immovable = immobileBase || (targetArchetype == RobotArchetype.ManipulatorArm);
                    _rootBody = ab;
                }
                else
                {
                    ab.jointType = ArticulationJointType.RevoluteJoint;
                    ab.anchorRotation = Quaternion.Euler(spec.AxisRotationEuler);
                    ab.twistLock = ArticulationDofLock.LimitedMotion;

                    var drive = ab.xDrive;
                    drive.lowerLimit = spec.LowerLimitDeg;
                    drive.upperLimit = spec.UpperLimitDeg;
                    drive.stiffness = spec.DriveStiffness;
                    drive.damping = spec.DriveDamping;
                    drive.forceLimit = spec.MaxForceLimit;
                    ab.xDrive = drive;

                    _actuatedJoints.Add(ab);
                }

                createdBodies.Add(ab);
            }
        }

        /// <summary>
        /// Reads the ground-truth generalized joint positions q* (radians) across all actuated joints.
        /// </summary>
        public float[] GetJointPositions()
        {
            float[] q = new float[_actuatedJoints.Count];
            for (int i = 0; i < _actuatedJoints.Count; i++)
            {
                ArticulationReducedSpace pos = _actuatedJoints[i].jointPosition;
                q[i] = pos.dofCount > 0 ? pos[0] : 0f;
            }
            return q;
        }

        /// <summary>
        /// Reads the ground-truth generalized joint angular velocities p* (rad/s) across all actuated joints.
        /// </summary>
        public float[] GetJointVelocities()
        {
            float[] p = new float[_actuatedJoints.Count];
            for (int i = 0; i < _actuatedJoints.Count; i++)
            {
                ArticulationReducedSpace vel = _actuatedJoints[i].jointVelocity;
                p[i] = vel.dofCount > 0 ? vel[0] : 0f;
            }
            return p;
        }

        /// <summary>
        /// Resets the root and all child ArticulationBody states deterministically for a given trajectory seed.
        /// </summary>
        public void ResetToDeterministicPose(int seed, float perturbationScale = 0.1f)
        {
            System.Random rng = new System.Random(seed);
            if (_rootBody != null)
            {
                Vector3 rootPos = new Vector3(0f, 1.05f, 0f);
                Quaternion rootRot = Quaternion.identity;
                _rootBody.TeleportRoot(rootPos, rootRot);
                _rootBody.velocity = Vector3.zero;
                _rootBody.angularVelocity = Vector3.zero;
            }

            for (int i = 0; i < _actuatedJoints.Count; i++)
            {
                float q0 = (float)(rng.NextDouble() * 2.0 - 1.0) * perturbationScale;
                float p0 = (float)(rng.NextDouble() * 2.0 - 1.0) * (perturbationScale * 0.5f);
                _actuatedJoints[i].jointPosition = new ArticulationReducedSpace(q0);
                _actuatedJoints[i].jointVelocity = new ArticulationReducedSpace(p0);
            }
        }

        private List<ArticulatedLinkSpec> GetArchetypeSpecs(RobotArchetype type)
        {
            var specs = new List<ArticulatedLinkSpec>();
            if (type == RobotArchetype.Humanoid)
            {
                // Root Pelvis + Torso + Left/Right Hip, Knee, Ankle, Shoulder (8 actuated DOF canonical humanoid)
                specs.Add(new ArticulatedLinkSpec
                {
                    Name = "Pelvis_Root", ParentIndex = -1,
                    LocalAnchorPosition = new Vector3(0f, 1.05f, 0f),
                    LinkDimensions = new Vector3(0.28f, 0.20f, 0.20f),
                    Mass = 12.0f, JointFriction = 0.05f, AngularDamping = 0.15f
                });
                string[] sideNames = { "Left", "Right" };
                float[] sideSigns = { -1f, 1f };

                for (int s = 0; s < 2; s++)
                {
                    float sign = sideSigns[s];
                    // Hip Pitch
                    specs.Add(new ArticulatedLinkSpec
                    {
                        Name = $"{sideNames[s]}_Hip", ParentIndex = 0,
                        LocalAnchorPosition = new Vector3(sign * 0.12f, -0.15f, 0f),
                        LinkDimensions = new Vector3(0.12f, 0.25f, 0.12f),
                        Mass = 4.5f, JointFriction = 0.08f, AngularDamping = 0.18f,
                        LowerLimitDeg = -60f, UpperLimitDeg = 75f,
                        DriveStiffness = 180f, DriveDamping = 22f, MaxForceLimit = 250f,
                        AxisRotationEuler = new Vector3(0f, 90f, 0f)
                    });
                    int hipIdx = specs.Count - 1;

                    // Knee Pitch
                    specs.Add(new ArticulatedLinkSpec
                    {
                        Name = $"{sideNames[s]}_Knee", ParentIndex = hipIdx,
                        LocalAnchorPosition = new Vector3(0f, -0.35f, 0f),
                        LinkDimensions = new Vector3(0.10f, 0.24f, 0.10f),
                        Mass = 3.0f, JointFriction = 0.08f, AngularDamping = 0.18f,
                        LowerLimitDeg = -110f, UpperLimitDeg = 5f,
                        DriveStiffness = 160f, DriveDamping = 20f, MaxForceLimit = 220f,
                        AxisRotationEuler = new Vector3(0f, 90f, 0f)
                    });
                    int kneeIdx = specs.Count - 1;

                    // Ankle Pitch
                    specs.Add(new ArticulatedLinkSpec
                    {
                        Name = $"{sideNames[s]}_Ankle", ParentIndex = kneeIdx,
                        LocalAnchorPosition = new Vector3(0f, -0.32f, 0.04f),
                        LinkDimensions = new Vector3(0.10f, 0.06f, 0.22f),
                        Mass = 1.2f, JointFriction = 0.10f, AngularDamping = 0.22f,
                        LowerLimitDeg = -45f, UpperLimitDeg = 45f,
                        DriveStiffness = 120f, DriveDamping = 16f, MaxForceLimit = 150f,
                        AxisRotationEuler = new Vector3(0f, 90f, 0f)
                    });

                    // Shoulder Pitch
                    specs.Add(new ArticulatedLinkSpec
                    {
                        Name = $"{sideNames[s]}_Shoulder", ParentIndex = 0,
                        LocalAnchorPosition = new Vector3(sign * 0.18f, 0.22f, 0f),
                        LinkDimensions = new Vector3(0.09f, 0.22f, 0.09f),
                        Mass = 2.0f, JointFriction = 0.06f, AngularDamping = 0.15f,
                        LowerLimitDeg = -75f, UpperLimitDeg = 75f,
                        DriveStiffness = 100f, DriveDamping = 14f, MaxForceLimit = 120f,
                        AxisRotationEuler = new Vector3(0f, 90f, 0f)
                    });
                }
            }
            else if (type == RobotArchetype.Quadruped)
            {
                specs.Add(new ArticulatedLinkSpec
                {
                    Name = "Trunk_Root", ParentIndex = -1,
                    LocalAnchorPosition = new Vector3(0f, 0.65f, 0f),
                    LinkDimensions = new Vector3(0.30f, 0.15f, 0.55f),
                    Mass = 14.0f, JointFriction = 0.06f, AngularDamping = 0.15f
                });
                string[] legNames = { "FL", "FR", "HL", "HR" };
                Vector3[] offsets =
                {
                    new Vector3(-0.15f, -0.08f, 0.22f),
                    new Vector3(0.15f, -0.08f, 0.22f),
                    new Vector3(-0.15f, -0.08f, -0.22f),
                    new Vector3(0.15f, -0.08f, -0.22f)
                };
                for (int l = 0; l < 4; l++)
                {
                    specs.Add(new ArticulatedLinkSpec
                    {
                        Name = $"{legNames[l]}_Thigh", ParentIndex = 0,
                        LocalAnchorPosition = offsets[l],
                        LinkDimensions = new Vector3(0.08f, 0.20f, 0.08f),
                        Mass = 2.2f, JointFriction = 0.08f, AngularDamping = 0.18f,
                        LowerLimitDeg = -60f, UpperLimitDeg = 60f,
                        DriveStiffness = 150f, DriveDamping = 18f, MaxForceLimit = 180f,
                        AxisRotationEuler = new Vector3(0f, 90f, 0f)
                    });
                    int thighIdx = specs.Count - 1;
                    specs.Add(new ArticulatedLinkSpec
                    {
                        Name = $"{legNames[l]}_Calf", ParentIndex = thighIdx,
                        LocalAnchorPosition = new Vector3(0f, -0.25f, 0f),
                        LinkDimensions = new Vector3(0.07f, 0.20f, 0.07f),
                        Mass = 1.4f, JointFriction = 0.08f, AngularDamping = 0.18f,
                        LowerLimitDeg = -90f, UpperLimitDeg = 10f,
                        DriveStiffness = 130f, DriveDamping = 16f, MaxForceLimit = 160f,
                        AxisRotationEuler = new Vector3(0f, 90f, 0f)
                    });
                }
            }
            else
            {
                // 8-DOF Serial Manipulator Arm
                specs.Add(new ArticulatedLinkSpec
                {
                    Name = "Base_Root", ParentIndex = -1,
                    LocalAnchorPosition = Vector3.zero,
                    LinkDimensions = new Vector3(0.25f, 0.12f, 0.25f),
                    Mass = 20.0f, JointFriction = 0.05f, AngularDamping = 0.15f
                });
                for (int j = 0; j < 8; j++)
                {
                    specs.Add(new ArticulatedLinkSpec
                    {
                        Name = $"Arm_Joint_{j}", ParentIndex = j,
                        LocalAnchorPosition = new Vector3(0f, 0.20f, 0f),
                        LinkDimensions = new Vector3(0.08f, 0.15f, 0.08f),
                        Mass = Mathf.Max(0.8f, 3.0f - 0.25f * j),
                        JointFriction = 0.08f, AngularDamping = 0.20f,
                        LowerLimitDeg = -120f, UpperLimitDeg = 120f,
                        DriveStiffness = 140f, DriveDamping = 20f, MaxForceLimit = 180f,
                        AxisRotationEuler = (j % 2 == 0) ? new Vector3(0f, 90f, 0f) : new Vector3(90f, 0f, 0f)
                    });
                }
            }
            return specs;
        }
    }
}
