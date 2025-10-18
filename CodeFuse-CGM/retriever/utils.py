import networkx as nx

def codegraph_to_nxgraph(graph):
    """
    将 CodeGraph 对象 转为 networkx 图对象
    :param graph: CodeGraph 对象
    :return graph_nx: nx.MultiDiGraph 对象
    """

    # 创建图 有向且允许两个节点之间有多条边
    G = nx.MultiDiGraph()

    # 增加点
    for node in graph.nodes:
        G.add_node(graph.nodes[node])

    # 增加边
    for edge in graph.edges:
        
        src_id = edge.source
        tgt_id = edge.target
        # 当前的数据中 会出现 不存在 node list 里的节点
        # 这样的 边 和 点 都先移除
        try:
            G.add_edge(graph.nodes[src_id], graph.nodes[tgt_id], type=edge.edge_type)
        except:
            pass
        
    print(f"nx Graph data parsed, nodes: {G.number_of_nodes():,}, edges: {G.number_of_edges():,}")
        
    return G

def codegraph_to_nxgraph_lite(graph):
    """
    轻量版 将 CodeGraph 对象 转为 networkx 图对象
    节点和边类型都脱离parser定义
    :param graph: CodeGraph 对象
    :return graph_nx: nx.MultiDiGraph 对象
    """
    
    # 创建图 有向且允许两个节点之间有多条边
    G = nx.MultiDiGraph()

    # 增加点
    for node in graph.nodes:
        G.add_node(node.node_id)

    # 增加边
    for edge in graph.edges:
        
        src_id = edge.source
        tgt_id = edge.target
        try:
            # 当前的数据中 会出现 不存在 node list 里的节点
            # 这样的 边 和 点 都先移除
            graph.nodes[src_id]
            graph.nodes[tgt_id]
            # 确保 源节点目标节点 都存在后，再加入
            G.add_edge(src_id, tgt_id, type=edge.edge_type.name)
        except:
            pass
        
    print(f"nx Graph lite data parsed, nodes: {G.number_of_nodes():,}, edges: {G.number_of_edges():,}")
        
    return G

def codegraph_to_nxgraph_analysis(graph):
    """
    分析版 将 CodeGraph 对象 转为 networkx 图对象
    专门用于路径分析的版本，移除 Repo 和 Package 节点；
    返回 有向图 和 无向图 两个 版本
    - 有向图：用于分析节点之间的转移概率；两个点之间可能存在多条边
    - 无向图：用于分析节点(File)之间的最短路径
    :param graph: CodeGraph 对象
    :return graph_nx: nx.MultiDiGraph 对象
    """
    
    # 创建图 有向且允许两个节点之间有多条边
    G_d = nx.MultiDiGraph()
    G_u = nx.Graph()

    # 增加点
    for node_id in graph.nodes:
        node = graph.get_node_by_id(node_id)
        node_type = node.get_type().name
        if node_type in ['REPO', 'PACKAGE']:
            continue
        G_d.add_node(node.node_id)
        G_u.add_node(node.node_id)

    # 增加边
    for edge in graph.edges:
        
        src_id = edge.source
        tgt_id = edge.target
        
        if G_d.has_node(src_id) and G_d.has_node(tgt_id):
            G_d.add_edge(src_id, tgt_id, type=edge.edge_type.name)
            G_u.add_edge(src_id, tgt_id, type=edge.edge_type.name)
        
    print(f"nx Graph analysis data parsed, nodes: {G_d.number_of_nodes():,}, edges: {G_d.number_of_edges():,}")
        
    return G_d, G_u