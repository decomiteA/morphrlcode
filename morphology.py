"""Generate parameterized ant XML files for morphology experiments.

Usage:
    python morphology.py --name long_legs --upper-leg 0.3 --lower-leg 0.6
    python morphology.py --name elongated --body-length 0.4
    python morphology.py --name fat_torso --torso-radius 0.35

Output: saves the XML to morphologies/<name>.xml
"""

import argparse
import math
import os
import mujoco


# ── Baseline values (standard ant) ────────────────────────────────────────────
BASELINE = dict(
    torso_radius  = 0.25,
    body_length   = 0.0,    # half-length of capsule body along x (0 = sphere)
    leg_radius    = 0.08,
    upper_leg     = 0.2,    # hip segment per-axis offset at 45°
    lower_leg     = 0.4,    # ankle segment per-axis offset at 45°
    gear          = 150.0,
)

_DIR = os.path.dirname(os.path.abspath(__file__))


def make_ant_xml(
    torso_radius: float = BASELINE["torso_radius"],
    body_length:  float = BASELINE["body_length"],
    leg_radius:   float = BASELINE["leg_radius"],
    upper_leg:    float = BASELINE["upper_leg"],
    lower_leg:    float = BASELINE["lower_leg"],
) -> str:
    """Return an ant XML string with the given morphology parameters.

    body_length > 0 changes the torso from a sphere to a capsule extending
    ±body_length along the x-axis (movement direction). Front legs attach at
    the +x end, back legs at the -x end.

    Gear is auto-scaled to maintain torque-to-weight ratio vs baseline.
    """

    # ── Auto-scale gear from baseline ─────────────────────────────────────────
    # Build the reference from BASELINE parameters (sphere ant) so gear scaling
    # stays consistent regardless of what ant.xml happens to contain.
    baseline_xml = _build_xml(
        BASELINE["torso_radius"], BASELINE["body_length"],
        BASELINE["leg_radius"], BASELINE["upper_leg"], BASELINE["lower_leg"],
        gear=BASELINE["gear"],
    )
    base_mj = mujoco.MjModel.from_xml_string(baseline_xml)

    raw_xml = _build_xml(torso_radius, body_length, leg_radius, upper_leg, lower_leg,
                         gear=BASELINE["gear"])
    new_mj = mujoco.MjModel.from_xml_string(raw_xml)

    # Scale gear by total body mass (torso + limbs) and leg length vs baseline
    base_mass    = sum(base_mj.body_mass)
    new_mass     = sum(new_mj.body_mass)
    length_ratio = (upper_leg + lower_leg) / (BASELINE["upper_leg"] + BASELINE["lower_leg"])
    mass_ratio   = new_mass / base_mass if base_mass > 0 else 1.0
    gear         = round(BASELINE["gear"] * mass_ratio * length_ratio, 1)

    return _build_xml(torso_radius, body_length, leg_radius, upper_leg, lower_leg, gear=gear)


def _build_xml(torso_radius, body_length, leg_radius, ul, ll, gear) -> str:
    """Construct the XML string from raw parameters."""

    BL = body_length  # half-length of body capsule along x

    # Torso geom: sphere if BL==0, capsule along x-axis otherwise
    if BL > 0:
        torso_geom = (f'<geom name="torso_geom" '
                      f'fromto="-{BL} 0 0 {BL} 0 0" '
                      f'size="{torso_radius}" type="capsule"/>')
    else:
        torso_geom = f'<geom name="torso_geom" pos="0 0 0" size="{torso_radius}" type="sphere"/>'

    # Front legs attach at +BL along x, back legs at -BL
    # Leg geometry/directions unchanged relative to their attachment point
    front_pos = f"{BL} 0 0"
    back_pos  = f"-{BL} 0 0"

    # init_z: torso radius is the vertical clearance (capsule is horizontal)
    init_z = round(ll * math.sqrt(2) * 0.6 + torso_radius, 3)

    return f"""\
<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.01" iterations="4" />
  <custom>
    <numeric data="0.0 0.0 {init_z} 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
    <numeric data="1000" name="constraint_limit_stiffness"/>
    <numeric data="4000" name="constraint_stiffness"/>
    <numeric data="10"   name="constraint_ang_damping"/>
    <numeric data="20"   name="constraint_vel_damping"/>
    <numeric data="0.5"  name="joint_scale_pos"/>
    <numeric data="0.2"  name="joint_scale_ang"/>
    <numeric data="0.0"  name="ang_damping"/>
    <numeric data="1"    name="spring_mass_scale"/>
    <numeric data="1"    name="spring_inertia_scale"/>
    <numeric data="15"   name="solver_maxls"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom contype="0" conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5"
     rgba="0.4 0.33 0.26 1.0"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" size="500 500 500" type="plane" rgba="0.5 0.5 0.5 1.0"/>
    <body name="torso" pos="0 0 {init_z}">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      {torso_geom}
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="{front_pos}">
        <geom fromto="0 0 0 {ul} {ul} 0" name="aux_1_geom" size="{leg_radius}" type="capsule"/>
        <body name="aux_1" pos="{ul} {ul} 0">
          <joint axis="0 0 1" name="hip_1" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 {ul} {ul} 0" name="left_leg_geom" size="{leg_radius}" type="capsule"/>
          <body name="ankle_body_1" pos="{ul} {ul} 0">
            <joint axis="-1 1 0" name="ankle_1" pos="0 0 0" range="30 70" type="hinge"/>
            <geom fromto="0 0 0 {ll} {ll} 0" name="left_ankle_geom" size="{leg_radius}" type="capsule"/>
            <geom name="left_foot_geom" contype="1" pos="{ll} {ll} 0" size="{leg_radius}" type="sphere" mass="0"/>
            <site name="foot_site_1" pos="{ll} {ll} 0" size="0.02"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="{back_pos}">
        <geom fromto="0 0 0 -{ul} {ul} 0" name="aux_2_geom" size="{leg_radius}" type="capsule"/>
        <body name="aux_2" pos="-{ul} {ul} 0">
          <joint axis="0 0 1" name="hip_2" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 -{ul} {ul} 0" name="right_leg_geom" size="{leg_radius}" type="capsule"/>
          <body name="ankle_body_2" pos="-{ul} {ul} 0">
            <joint axis="1 1 0" name="ankle_2" pos="0 0 0" range="-70 -30" type="hinge"/>
            <geom fromto="0 0 0 -{ll} {ll} 0" name="right_ankle_geom" size="{leg_radius}" type="capsule"/>
            <geom name="right_foot_geom" contype="1" pos="-{ll} {ll} 0" size="{leg_radius}" type="sphere" mass="0"/>
            <site name="foot_site_2" pos="-{ll} {ll} 0" size="0.02"/>
          </body>
        </body>
      </body>
      <body name="back_leg" pos="{back_pos}">
        <geom fromto="0 0 0 -{ul} -{ul} 0" name="aux_3_geom" size="{leg_radius}" type="capsule"/>
        <body name="aux_3" pos="-{ul} -{ul} 0">
          <joint axis="0 0 1" name="hip_3" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 -{ul} -{ul} 0" name="back_leg_geom" size="{leg_radius}" type="capsule"/>
          <body name="ankle_body_3" pos="-{ul} -{ul} 0">
            <joint axis="-1 1 0" name="ankle_3" pos="0 0 0" range="-70 -30" type="hinge"/>
            <geom fromto="0 0 0 -{ll} -{ll} 0" name="third_ankle_geom" size="{leg_radius}" type="capsule"/>
            <geom name="third_foot_geom" contype="1" pos="-{ll} -{ll} 0" size="{leg_radius}" type="sphere" mass="0"/>
            <site name="foot_site_3" pos="-{ll} -{ll} 0" size="0.02"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="{front_pos}">
        <geom fromto="0 0 0 {ul} -{ul} 0" name="aux_4_geom" size="{leg_radius}" type="capsule"/>
        <body name="aux_4" pos="{ul} -{ul} 0">
          <joint axis="0 0 1" name="hip_4" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 {ul} -{ul} 0" name="rightback_leg_geom" size="{leg_radius}" type="capsule"/>
          <body name="ankle_body_4" pos="{ul} -{ul} 0">
            <joint axis="1 1 0" name="ankle_4" pos="0 0 0" range="30 70" type="hinge"/>
            <geom fromto="0 0 0 {ll} -{ll} 0" name="fourth_ankle_geom" size="{leg_radius}" type="capsule"/>
            <geom name="fourth_foot_geom" contype="1" pos="{ll} -{ll} 0" size="{leg_radius}" type="sphere" mass="0"/>
            <site name="foot_site_4" pos="{ll} -{ll} 0" size="0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4"   gear="{gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="{gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1"   gear="{gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="{gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2"   gear="{gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="{gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3"   gear="{gear}"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="{gear}"/>
  </actuator>
</mujoco>
"""


def parse_args():
    p = argparse.ArgumentParser(description="Generate parameterized ant XML")
    p.add_argument("--name",         required=True,        help="Output filename (saved to morphologies/<name>.xml)")
    p.add_argument("--torso-radius", type=float, default=BASELINE["torso_radius"])
    p.add_argument("--body-length",  type=float, default=BASELINE["body_length"],
                   help="Half-length of capsule body along x-axis (0 = sphere, default 0)")
    p.add_argument("--leg-radius",   type=float, default=BASELINE["leg_radius"])
    p.add_argument("--upper-leg",    type=float, default=BASELINE["upper_leg"],
                   help="Upper leg per-axis offset (baseline 0.2)")
    p.add_argument("--lower-leg",    type=float, default=BASELINE["lower_leg"],
                   help="Lower leg per-axis offset (baseline 0.4)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    xml = make_ant_xml(
        torso_radius = args.torso_radius,
        body_length  = args.body_length,
        leg_radius   = args.leg_radius,
        upper_leg    = args.upper_leg,
        lower_leg    = args.lower_leg,
    )

    out_dir = os.path.join(_DIR, "morphologies")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}.xml")

    with open(out_path, "w") as f:
        f.write(xml)

    print(f"Saved: {out_path}")

    mj = mujoco.MjModel.from_xml_string(xml)
    gear_val = mj.actuator_gear[0, 0]
    print(f"  torso_radius : {args.torso_radius}")
    print(f"  body_length  : {args.body_length}  (half-length, 0=sphere)")
    print(f"  leg_radius   : {args.leg_radius}")
    print(f"  upper_leg    : {args.upper_leg}")
    print(f"  lower_leg    : {args.lower_leg}")
    print(f"  gear (auto)  : {gear_val:.1f}")
