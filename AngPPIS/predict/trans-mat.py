import numpy as np
import os
from Bio.PDB import PDBParser
import math
from tqdm import tqdm

def calc_angle(v1, v2, v3):
    """计算由v1, v2, v3定义的角度，v2是顶点。"""
    a = np.array(v1)
    b = np.array(v2)
    c = np.array(v3)
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product == 0:
        return 0
    cosine_angle = np.dot(ba, bc) / norm_product
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    return np.degrees(angle)

def compute_avg_pos(residue):
    """计算氨基酸中原子的平均位置"""
    coords = [atom.get_coord() for atom in residue.get_atoms() if atom.get_name() != 'H']
    if coords:
        avg_pos = np.mean(coords, axis=0)
        return avg_pos
    else:
        return None

def generate_angle_matrices_for_pdb(pdb_file, output_dir):
    """为单个PDB文件生成角度矩阵并保存"""
    parser = PDBParser()
    structure = parser.get_structure(pdb_file, pdb_file)
    model = structure[0]
    residues = [residue for residue in model.get_residues() if residue.get_resname() not in ['HOH', 'WAT']]
    avg_positions = [compute_avg_pos(residue) for residue in residues]
    
    for i, pos in enumerate(avg_positions):
        if pos is None:
            continue
        angles_matrix = np.zeros((len(avg_positions), len(avg_positions)))
        for j, pos_j in enumerate(avg_positions):
            for k, pos_k in enumerate(avg_positions):
                if pos_j is None or pos_k is None:
                    continue
                if j == k:  # 忽略与自身的角度
                    continue
                angle = calc_angle(pos_j, pos, pos_k)
                angles_matrix[j, k] = angle
        
        # 删除与当前氨基酸自身相关的行和列，去除十字交叉图案
        angles_matrix = np.delete(angles_matrix, i, axis=0)  # 删除行
        angles_matrix = np.delete(angles_matrix, i, axis=1)  # 删除列

        # 保存矩阵为.npy文件
        filename = f"{os.path.splitext(os.path.basename(pdb_file))[0]}_{i}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, angles_matrix)

def batch_process_pdb_files(pdb_dir, output_dir):
    """批量处理pdb_dir中的所有PDB文件"""
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        generate_angle_matrices_for_pdb(pdb_path, output_dir)

# 示例使用
pdb_dir = 'pdb'  # 指向你的PDB文件夹
output_dir = 'input'  # 输出目录
batch_process_pdb_files(pdb_dir, output_dir)
