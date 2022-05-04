import multiprocessing as mlp

import networkx as nx

from BasicMethods import *


class NodeCollapse(BasicMethods):
    def __init__(self, g):
        super(NodeCollapse, self).__init__("NodeCollapse")
        self.G = nx.Graph(g)
        self.G.remove_edges_from(nx.selfloop_edges(self.G)) # 删去自连边
        self.kshell = self.kshellDecomposition(self.G)      # kshell分布 节点:k-value的键值对
        self.kmax = max(self.kshell.values())               # 最大kshell
        self.core = nx.k_core(self.G, self.kmax)            # 提取kmxax-core

    """计算删除图中连边后，导致坍塌的节点"""
    def collapsedNodes(self, graph, deletedNode, kmax):
        g = graph.copy()
        coreNodes = set(g.nodes())
        g.remove_node(deletedNode)
        _core = nx.k_core(g, kmax)
        _coreNodes = set(_core.nodes())
        return list(coreNodes - _coreNodes)

    def HDN(self, graph):
        g = graph.copy()
        maxWeight = 0
        degree = dict(g.degree)
        for node in g.nodes:
            weight = degree[node]
            if weight > maxWeight:
                maxWeight = weight
                self.stack.clear()
                self.stack.push(node)
            elif weight == maxWeight:
                self.stack.push(node)
        collapsedNodes = self.stack.clear()
        return collapsedNodes

    def CKC(self, graph):
        g = graph.copy()
        # corona = nx.k_corona(G=g, k=self.kmax)
        nodes = list(nx.k_corona(G=g, k=self.kmax).nodes())
        temp = nodes.copy()
        for node in nodes:
            neighbors = [n for n in g[node]]
            temp.extend(neighbors)
        candidates = list(set(temp))
        deletedNodes = candidates.copy()
        for node in candidates:
            if node in deletedNodes:
                followers = self.collapsedNodes(g, node, self.kmax)
                followers.remove(node)
                deletedNodes = list(set(deletedNodes) - set(followers))
            else:
                continue
        return deletedNodes

    def nodeAttack(self, strategy=None):
        g = self.G.copy()
        core = self.core.copy()
        notEmpty, deletedNodeNum = True, 0
        while notEmpty:
            if strategy == "CKC":
                collapsedNodes = self.CKC(core)
            else:
                collapsedNodes = self.HDN(core)
            g.remove_nodes_from(collapsedNodes)
            core.remove_nodes_from(collapsedNodes)
            core = nx.k_core(core, self.kmax)
            notEmpty = True if core else False
            deletedNodeNum += len(collapsedNodes)
        return [deletedNodeNum, g]


def nodeAttack(strategy="HDN"):
    dirs = os.listdir(data_path)
    dirs.sort(key=lambda x: int(x[0: 2]))
    df = pd.DataFrame(columns=['|V|', '|E|', 'kmax', 'knode', 'klink', 'NDN', 'NDE', 'ECR%', 'ASR%', 'FAR%', 'Time'],
                      index=[i.split('.')[0] for i in dirs])
    for fullname in dirs:
        filename = fullname.split(".")[0]
        file = open(os.path.join(data_path, fullname), "rb")
        g = NodeCollapse(nx.Graph(pkl.load(file)))
        # 获取基本信息
        edgeNum = len(g.G.edges)
        nodeNum = len(g.G.nodes)
        k_edgeNum = len(g.core.edges)
        k_nodeNum = len(g.core.nodes)
        ticks = time.process_time()
        results = g.nodeAttack(strategy)
        ticks = time.process_time() - ticks
        print("%s: %s: %.4fs" % (strategy, filename, ticks))
        deletedEdgeNum = len(g.G.edges) - len(results[1].edges)
        _kshell = g.kshellDecomposition(results[1])
        ASR = 1 - g.attackAccuracy(g.kshell, _kshell)
        spillRate = ASR - k_nodeNum / nodeNum
        FAR = spillRate * nodeNum / (nodeNum - k_nodeNum)

        df.loc[filename] = [nodeNum, edgeNum, g.kmax,
                             k_nodeNum, k_edgeNum,
                             results[0], deletedEdgeNum, round(deletedEdgeNum / edgeNum * 100, 4),
                             round(ASR * 100, 4), round(FAR * 100, 4),
                             round(ticks, 4)]
        file.close()
        print(df.loc[filename])
    df.to_excel('./cache/excel/coreAttack_' + str(strategy) + '.xlsx')
    print(df)


if __name__ == "__main__":
    p1 = mlp.Process(target=nodeAttack, args=())
    p2 = mlp.Process(target=nodeAttack, args=("CKC",))
    p1.start()
    p2.start()

    pass