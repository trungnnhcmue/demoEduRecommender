import pandas as pd
import networkx as nx
from pyvis.network import Network
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

def build_graph():
    students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    courses = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    enrollments = pd.read_csv(os.path.join(DATA_DIR, "enrollments.csv"))

    G = nx.Graph()

    # --- Node ---
    for _, row in students.iterrows():
        G.add_node(f"student_{row['student_id']}", 
                   label=row['name'], 
                   type="student", 
                   major=row['major'], 
                   year=row['year'])
    
    for _, row in courses.iterrows():
        G.add_node(f"course_{row['course_id']}", 
                   label=row['course_name'], 
                   type="course", 
                   major=row['major'], 
                   credits=row['credits'])
        if not pd.isna(row['prerequisite']):
            prereq = int(row['prerequisite'])
            if prereq < row['course_id']:
                G.add_edge(f"course_{prereq}", f"course_{row['course_id']}", relation="prerequisite")

    for major in students["major"].unique():
        G.add_node(f"major_{major}", label=major, type="major")

    # --- Edges ---
    for _, row in enrollments.iterrows():
        sid = f"student_{row['student_id']}"
        cid = f"course_{row['course_id']}"
        G.add_edge(sid, cid, relation="enrolled_in", semester=row["semester"], grade=row["grade"])
    
    for _, row in students.iterrows():
        G.add_edge(f"student_{row['student_id']}", f"major_{row['major']}", relation="has_major")
    
    for _, row in courses.iterrows():
        G.add_edge(f"course_{row['course_id']}", f"major_{row['major']}", relation="belongs_to_major")

    return G


def visualize_student_graph(G, student_id, output_path="student_graph.html", max_nodes=20):
    """
    Hiển thị đồ thị con cho một sinh viên.
    Chỉ hiển thị tối đa `max_nodes` node liên quan để gọn.
    """
    student_node = f"student_{student_id}"
    if student_node not in G:
        raise ValueError(f"Không tìm thấy sinh viên có ID = {student_id}")

    # --- Lấy các node theo khoảng cách (bfs) ---
    distances = nx.single_source_shortest_path_length(G, student_node)
    # sắp xếp theo khoảng cách, ưu tiên gần sinh viên
    sorted_nodes = sorted(distances.items(), key=lambda x: (x[1], x[0]))
    # lấy tối đa max_nodes
    selected_nodes = [node for node, dist in sorted_nodes[:max_nodes]]

    # --- Lấy các node "neighbors trực tiếp" trong top max_nodes ---
    subG = G.subgraph(selected_nodes)

    # --- Pyvis visualization ---
    net = Network(height="600px", width="90%", bgcolor="#ffffff", font_color="black", notebook=False)
    color_map = {"student": "blue", "course": "green", "major": "red"}

    for node, data in subG.nodes(data=True):
        ntype = data.get("type", "other")
        net.add_node(
            node,
            label=data.get("label", node),
            color=color_map.get(ntype, "gray"),
            size=15 if ntype=="student" else 10,
            title=f"{data.get('label', node)} ({ntype})"
        )

    for u, v, data in subG.edges(data=True):
        net.add_edge(u, v, label=data.get("relation", ""), width=1)

    # tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net.write_html(output_path, notebook=False, open_browser=False)
    print(f"Đã tạo đồ thị nhỏ cho sinh viên {student_id}: {output_path}")


if __name__ == "__main__":
    G = build_graph()
    visualize_student_graph(G, student_id=1, output_path="student_graph.html")
