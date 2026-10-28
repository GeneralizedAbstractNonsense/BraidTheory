import os
import numpy as np


class B4BraidGenerator:

    def __init__(
        self,
        total_length=297.0,
        total_width=190.0,
        strand_radius=8.0,
        height_factor=0.5,
    ):
        """Initialize parameters for the B4 Braid Generator.

        Units are in millimeters (mm).
        """
        self.total_length = total_length
        self.total_width = total_width
        self.strand_radius = strand_radius

        # Spacing between adjacent strand centers
        self.x_spacing = (self.total_width - 2 * self.strand_radius) / 3.0
        self.height_over = self.x_spacing * height_factor

        # Normalized RGB colors [0.0, 1.0] for MTL
        self.colors = {
            1: (0.0 / 255.0, 139.0 / 255.0, 0.0 / 255.0),  # Green: RGB(0,139,0)
            2: (22.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0),  # Blue: RGB(22,0,255)
            3: (
                253.0 / 255.0,
                255.0 / 255.0,
                0.0 / 255.0,
            ),  # Yellow: RGB(253,255,0)
            4: (255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0),  # Red: RGB(255,0,0)
        }

        # Initial X coordinates centered around X=0
        half_w = (3 * self.x_spacing) / 2.0
        self.initial_x = [
            -half_w,
            -half_w + self.x_spacing,
            -half_w + 2 * self.x_spacing,
            -half_w + 3 * self.x_spacing,
        ]

    def _generate_strand_paths(self, braid_word, steps_per_gen=30):
        """Generates 3D trajectories for each of the 4 strands."""
        paths = {m: [] for m in range(1, 5)}
        current_slots = [1, 2, 3, 4]

        # Starting point at Y = 0
        for slot_idx, strand_id in enumerate(current_slots):
            paths[strand_id].append([self.initial_x[slot_idx], 0.0, 0.0])

        # If braid_word is empty, generate identity braid (straight parallel strands)
        if not braid_word:
            t_vals = np.linspace(0, 1, steps_per_gen)[1:]
            for t in t_vals:
                y = t * self.total_length
                for slot_idx, s_id in enumerate(current_slots):
                    paths[s_id].append([self.initial_x[slot_idx], y, 0.0])
            return {s_id: np.array(pts) for s_id, pts in paths.items()}

        num_generators = len(braid_word)
        step_length = self.total_length / num_generators
        current_y = 0.0

        for gen in braid_word:
            gen_clean = gen.lower().strip()
            if not gen_clean:
                continue

            is_inverse = "inv" in gen_clean
            sigma_num = int(gen_clean.replace("sigma", "").replace("inv", ""))

            idx1 = sigma_num - 1
            idx2 = sigma_num

            strand_a = current_slots[idx1]
            strand_b = current_slots[idx2]

            x1_start, x2_start = self.initial_x[idx1], self.initial_x[idx2]
            x1_end, x2_end = x2_start, x1_start

            z_a_max = self.height_over if not is_inverse else -self.height_over
            z_b_max = -self.height_over if not is_inverse else self.height_over

            t_vals = np.linspace(0, 1, steps_per_gen)[1:]

            for t in t_vals:
                y = current_y + t * step_length
                factor = (1 - np.cos(np.pi * t)) / 2.0

                xa = x1_start + (x1_end - x1_start) * factor
                za = z_a_max * np.sin(np.pi * t)
                paths[strand_a].append([xa, y, za])

                xb = x2_start + (x2_end - x2_start) * factor
                zb = z_b_max * np.sin(np.pi * t)
                paths[strand_b].append([xb, y, zb])

                for slot_idx, s_id in enumerate(current_slots):
                    if s_id not in (strand_a, strand_b):
                        x_curr = self.initial_x[slot_idx]
                        paths[s_id].append([x_curr, y, 0.0])

            current_slots[idx1], current_slots[idx2] = strand_b, strand_a
            current_y += step_length

        return {s_id: np.array(pts) for s_id, pts in paths.items()}

    def _build_tube(self, curve, sides=16):
        """Constructs a 3D cylindrical mesh with OUTWARD facing normals."""
        num_pts = len(curve)
        verts, faces = [], []

        for i in range(num_pts):
            tangent = (
                curve[i + 1] - curve[i]
                if i < num_pts - 1
                else curve[i] - curve[i - 1]
            )
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 0:
                tangent /= tangent_norm

            up = (
                np.array([0, 0, 1])
                if abs(tangent[2]) < 0.9
                else np.array([1, 0, 0])
            )
            normal = np.cross(tangent, up)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 0:
                normal /= normal_norm
            binormal = np.cross(tangent, normal)

            for j in range(sides):
                angle = (2 * np.pi * j) / sides
                offset = (
                    np.cos(angle) * normal + np.sin(angle) * binormal
                ) * self.strand_radius
                verts.append(curve[i] + offset)

        for i in range(num_pts - 1):
            for j in range(sides):
                next_j = (j + 1) % sides
                v1 = i * sides + j + 1
                v2 = i * sides + next_j + 1
                v3 = (i + 1) * sides + j + 1
                v4 = (i + 1) * sides + next_j + 1

                # Counter-Clockwise (CCW) winding order -> Outward normals
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

        return np.array(verts), np.array(faces)

    def generate_files(self, braid_word, obj_filename):
        """Exports OBJ and MTL files for the generated 3D braid model."""
        base_name, _ = os.path.splitext(obj_filename)
        mtl_filename = f"{base_name}.mtl"
        obj_filename = f"{base_name}.obj"

        paths = self._generate_strand_paths(braid_word)

        # Write MTL File
        with open(mtl_filename, "w") as mtl:
            mtl.write("# B4 Braid Material Library\n")
            for strand_id, (r, g, b) in self.colors.items():
                mtl.write(f"newmtl mat_strand_{strand_id}\n")
                mtl.write(f"Kd {r:.5f} {g:.5f} {b:.5f}\n")
                mtl.write("Ks 0.2 0.2 0.2\nNs 100\n\n")

        # Write OBJ File
        with open(obj_filename, "w") as obj:
            obj.write(f"mtllib {mtl_filename}\n\n")
            v_offset = 0

            for strand_id in range(1, 5):
                curve = paths[strand_id]
                verts, faces = self._build_tube(curve)

                obj.write(f"o Strand_{strand_id}\n")
                obj.write(f"usemtl mat_strand_{strand_id}\n")

                for v in verts:
                    obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                for f in faces:
                    obj.write(
                        f"f {f[0] + v_offset} {f[1] + v_offset} {f[2] + v_offset}\n"
                    )

                v_offset += len(verts)

        print("\nB4 Braid Model generated successfully!")
        print(f" - OBJ File: {obj_filename}")
        print(f" - MTL File: {mtl_filename}")


if __name__ == "__main__":
    print("==========================================")
    print("        B4 BRAID 3D MODEL GENERATOR       ")
    print("==========================================\n")

    print("Enter braid generators separated by spaces or commas.")
    print("Valid generators: sigma1, sigma2, sigma3, sigma1inv, sigma2inv, sigma3inv")
    print("Leave empty and press ENTER to generate the identity braid (straight strands).\n")
    word_input = input("Braid Word: ").strip()

    if word_input:
        braid_word = [
            w.strip() for w in word_input.replace(",", " ").split() if w.strip()
        ]
    else:
        braid_word = []

    length_input = input("\nEnter total length in mm [Default: 297]: ").strip()
    total_length = float(length_input) if length_input else 297.0

    width_input = input("Enter total width in mm [Default: 190]: ").strip()
    total_width = float(width_input) if width_input else 190.0

    default_filename = "braid_identity" if not braid_word else "braid_model"
    filename_input = input(
        f"Enter output file name [Default: {default_filename}]: "
    ).strip()
    filename = filename_input if filename_input else default_filename

    generator = B4BraidGenerator(
        total_length=total_length, total_width=total_width
    )
    generator.generate_files(braid_word, filename)
