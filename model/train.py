import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model.utils import load_data, build_graph
from model.ftpcomplex import FTPComplex

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def train_and_eval(epochs=100, neg_ratio=5, lr=0.01, emb_dim=64, top_k=10):
    # ==== Load dữ liệu ====
    students, courses, enrollments = load_data(
        "data/students.csv", "data/courses.csv", "data/enrollments.csv"
    )

    # ==== Train/test split (positive edges) ====
    train_enroll, test_enroll = train_test_split(enrollments, test_size=0.2, random_state=42)

    # ==== Build graph từ train set ====
    data = build_graph(students, courses, train_enroll)

    # ==== Khởi tạo model ====
    model = FTPComplex(num_nodes=data.num_nodes, embedding_dim=emb_dim, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # ==== Positive edges train ====
    pos_edges = []
    for s, c in zip(train_enroll["student_id"], train_enroll["course_id"]):
        s_idx = data.student2idx[s]
        c_idx = data.course2idx[c] + data.num_students
        pos_edges.append((s_idx, c_idx))
    pos_edges = torch.tensor(pos_edges, dtype=torch.long)

    # ==== Negative edges train ====
    neg_edges = []
    for s in train_enroll["student_id"].unique():
        for _ in range(neg_ratio):
            c = random.choice(courses["course_id"].values)
            if ((train_enroll["student_id"] == s) & (train_enroll["course_id"] == c)).any():
                continue
            s_idx = data.student2idx[s]
            c_idx = data.course2idx[c] + data.num_students
            neg_edges.append((s_idx, c_idx))
    neg_edges = torch.tensor(neg_edges, dtype=torch.long)

    # ==== Gộp edges + labels train ====
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    labels = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))])

    # ==== Training ====
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        emb = model.get_embedding(data.edge_index)
        scores = (emb[edges[:, 0]] * emb[edges[:, 1]]).sum(dim=1)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # ==== Evaluation (Recall@K) ====
    model.eval()
    with torch.no_grad():
        emb = model.get_embedding(data.edge_index)

    recall_scores = []
    for sid in test_enroll["student_id"].unique():
        s_idx = data.student2idx[sid]
        student_emb = emb[s_idx]

        # All courses chưa học trong train
        taken_courses = set(train_enroll[train_enroll["student_id"] == sid]["course_id"].values)
        candidate_scores = []
        for cid, idx in data.course2idx.items():
            if cid in taken_courses:
                continue
            c_idx = idx + data.num_students
            score = torch.dot(student_emb, emb[c_idx]).item()
            candidate_scores.append((cid, score))

        # Lấy top-K gợi ý
        top_courses = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:top_k]
        top_course_ids = {cid for cid, _ in top_courses}

        # Các course trong test thực sự
        true_courses = set(test_enroll[test_enroll["student_id"] == sid]["course_id"].values)

        if len(true_courses) > 0:
            recall = len(top_course_ids & true_courses) / len(true_courses)
            recall_scores.append(recall)

    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    print(f"Recall@{top_k}: {avg_recall:.4f}")

    # ==== Lưu model đã train ====
    os.makedirs("model/saved", exist_ok=True)
    torch.save(model.state_dict(), "model/saved/ftpcomplex.pt")

    return avg_recall


if __name__ == "__main__":
    train_and_eval(epochs=100, top_k=5)
