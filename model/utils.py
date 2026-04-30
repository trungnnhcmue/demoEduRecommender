import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

def load_data(students_path, courses_path, enrollments_path):
    students = pd.read_csv(students_path)
    courses = pd.read_csv(courses_path)
    enrollments = pd.read_csv(enrollments_path)
    return students, courses, enrollments

def build_graph(students, courses, enrollments):
    """
    Xây dựng đồ thị 2 lớp: student - course, có node features
    """
    num_students = len(students)
    num_courses = len(courses)

    # ==== Gán id index cho students và courses ====
    student2idx = {sid: i for i, sid in enumerate(students["student_id"].values)}
    course2idx = {cid: i for i, cid in enumerate(courses["course_id"].values)}

    # ==== Tạo edge list (student -> course) ====
    edges = []
    for _, row in enrollments.iterrows():
        s_idx = student2idx[row["student_id"]]
        c_idx = course2idx[row["course_id"]] + num_students  # offset cho course
        edges.append([s_idx, c_idx])
        edges.append([c_idx, s_idx])  # undirected

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ==== Node features ====
    # Encode major cho student và course
    majors = list(set(students["major"].unique()) | set(courses["major"].unique()))
    major2idx = {m: i for i, m in enumerate(majors)}

    # Student features: [one-hot major] + [year/4]
    student_major = [major2idx[m] for m in students["major"]]
    student_major_feat = F.one_hot(torch.tensor(student_major), num_classes=len(majors))
    student_year_feat = torch.tensor(students["year"].values).view(-1, 1) / 4.0
    student_feat = torch.cat([student_major_feat.float(), student_year_feat.float()], dim=1)

    # Course features: [one-hot major] + [credits/4]
    course_major = [major2idx[m] for m in courses["major"]]
    course_major_feat = F.one_hot(torch.tensor(course_major), num_classes=len(majors))
    course_credit_feat = torch.tensor(courses["credits"].values).view(-1, 1) / 4.0
    course_feat = torch.cat([course_major_feat.float(), course_credit_feat.float()], dim=1)

    # Ghép student + course features
    x = torch.cat([student_feat, course_feat], dim=0)

    # ==== Tổng số node ====
    num_nodes = num_students + num_courses

    # ==== Build Data object ====
    data = Data(edge_index=edge_index, num_nodes=num_nodes, x=x)
    data.num_students = num_students
    data.num_courses = num_courses
    data.student2idx = student2idx
    data.course2idx = course2idx
    data.major2idx = major2idx

    return data
