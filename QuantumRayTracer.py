from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
import time

def vector_normalization(vector):
    return vector / np.linalg.norm(vector)

def intersection_with_sphere(center, radius, origin, direction_vector):
    b = 2 * np.dot(direction_vector, origin - center)
    c = np.linalg.norm(origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def closest_intersection(spheres, origin, direction):
    distances = [intersection_with_sphere(sphere['center'], sphere['radius'], origin, direction) for sphere in spheres]
    closest_sphere = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            closest_sphere = spheres[index]
    return closest_sphere, min_distance

def reflection_ray(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def quantum_supersampling(scene_value, num_qubits):
    qc = QuantumCircuit(num_qubits*2, num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    qc.barrier()
    qc.cx(list(range(num_qubits)), list(range(num_qubits, num_qubits*2)))
    grover_iterations = int(np.pi/4 * np.sqrt(2**num_qubits) * np.mean(scene_value))
    for _ in range(grover_iterations):
        qc.append(grover_iteration(num_qubits), list(range(num_qubits*2)))
    apply_inverse_qft(qc, num_qubits)
    qc.measure(range(num_qubits, num_qubits*2), range(num_qubits))
    return qc

def grover_iteration(num_qubits):
    qc = QuantumCircuit(num_qubits*2)
    qc.h(list(range(num_qubits)))
    qc.barrier()
    for i in range(num_qubits):
        qc.x(i)
    qc.cx(num_qubits-1, num_qubits)
    for i in range(num_qubits):
        qc.x(i)
    qc.barrier()
    qc.h(list(range(num_qubits)))
    qc.x(list(range(num_qubits)))
    qc.h(num_qubits-1)
    qc.mcp(np.pi, list(range(num_qubits-1)), num_qubits-1)
    qc.h(num_qubits-1)
    qc.x(list(range(num_qubits)))
    qc.h(list(range(num_qubits)))
    return qc

def apply_inverse_qft(qc, num_qubits):
    for j in range(num_qubits):
        qc.h(j)
        for k in range(j+1, num_qubits):
            qc.cp(-2 * np.pi / float(2 ** (k - j)), k, j)
    for qubit in range(num_qubits // 2):
        qc.swap(qubit, num_qubits - qubit - 1)

def estimate_pixel_value(counts, num_samples):
    total = 0
    divisor = 2 ** num_samples - 1
    shots = sum(counts.values())
    for key, value in counts.items():
        decimal_value = int(key, 2)
        scaled_value = decimal_value / divisor
        total += scaled_value * value
    return total / shots

spheres = [
    {'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'Diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'Diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'Diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5},
    {'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'Diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5}
]

light = {'position': np.array([5, 5, 5]), 'ambient': np.array([0.3, 0.3, 0.3]), 'Diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1])}

width = 300
height = 200
max_depth = 3
camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio)
image = np.zeros((height, width, 3))
num_qubits = 4
num_samples = 50

start_time = time.time()
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        print("Pixel:", str(i) + "/" + str(height), str(j) + "/" + str(width))
        color_accumulator = np.zeros((3))
        for _ in range(num_samples):
            random_offset = np.random.uniform(-0.5, 0.5, 2) / width
            pixel = np.array([x, y, 0]) + np.append(random_offset, 0)
            origin = camera
            direction = vector_normalization(pixel - origin)
            color = np.zeros((3))
            reflection = 1
            for _ in range(max_depth):
                closest_sphere, min_distance = closest_intersection(spheres, origin, direction)
                if closest_sphere is None:
                    break
                intersection = origin + min_distance * direction
                normal = vector_normalization(intersection - closest_sphere['center'])
                shifted_point = intersection + 1e-5 * normal
                to_light = vector_normalization(light['position'] - shifted_point)
                _, shadow_distance = closest_intersection(spheres, shifted_point, to_light)
                is_shadowed = shadow_distance < np.linalg.norm(light['position'] - intersection)
                illumination = np.zeros((3))
                if not is_shadowed:
                    diffuse_intensity = max(0, np.dot(to_light, normal))
                    specular_intensity = np.dot(vector_normalization(to_light + vector_normalization(camera - intersection)), normal)
                    specular_intensity = max(0, specular_intensity) ** closest_sphere['shininess']
                    illumination += (closest_sphere['Diffuse'] * light['Diffuse'] * diffuse_intensity +
                                     closest_sphere['specular'] * light['specular'] * specular_intensity)
                color += reflection * illumination
                reflection *= closest_sphere['reflection']
                origin = shifted_point
                direction = reflection_ray(direction, normal)
            qc = quantum_supersampling(color, num_qubits)
            simulator = AerSimulator()
            transpiled_qc = transpile(qc, simulator)
            qobj = assemble(transpiled_qc)
            result = simulator.run(qobj).result()
            counts = result.get_counts(transpiled_qc)
            estimated_value = estimate_pixel_value(counts, num_samples)
            color += estimated_value
            color_accumulator += np.clip(color, 0, 1)
        averaged_color = color_accumulator / num_samples
        print(averaged_color)
        image[i, j] = np.clip(averaged_color, 0, 1)
end_time = time.time()
total_time = end_time - start_time
print("Total runtime:", total_time, "seconds")
plt.imshow(image)
plt.show()