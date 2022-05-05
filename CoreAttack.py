import networkx as nx

from BasicMethods import *
import multiprocessing as mlp


class CoreAttack(BasicMethods):
    def __init__(self, g):
        super(CoreAttack, self).__init__("CoreAttack")
        self.G = nx.Graph(g)
        self.G.remove_edges_from(nx.selfloop_edges(self.G)) # 删去自连边
        self.kshell = self.kshellDecomposition(self.G)      # kshell分布 节点:k-value的键值对
        self.kmax = max(self.kshell.values())               # 最大kshell
        self.core = nx.k_core(self.G, self.kmax)            # 提取kmxax-core
        self.crossNum = 0
        self.totalcross = 0

    """基本功能函数"""
    """计算删除图中连边后，导致坍塌的节点"""
    def collapsedNodes(self, graph, deletedEdges, kmax):
        g = graph.copy()
        coreNodes = set(g.nodes())
        coreEdges = set(g.edges())
        g.remove_edges_from(deletedEdges)
        _core = nx.k_core(g, kmax)
        _coreNodes = set(_core.nodes())
        _coreEdges = set(_core.edges())
        return list(coreNodes - _coreNodes), list(coreEdges - _coreEdges)

    """使用贪婪策略对最大k-core进行坍塌攻击,即每次删除一条可以使得坍塌范围最大的连边"""
    def greedyCollapseCore(self, graph, kmax):
        g = graph.copy()
        coronaCore = nx.connected_components(nx.k_corona(G=g, k=kmax))
        collapsedEdges = []
        pastNodes = []
        for component in coronaCore:
            node = random.sample(list(component), 1)[0]
            edge = random.sample(list(g.edges(node)), 1)
            if node in pastNodes:
                continue
            elif len(component) <= 1:
                collapsedEdges.append([edge, 1, False])
                continue
            followers, _ = self.collapsedNodes(g, edge, kmax)
            crossFlag = False
            if len(followers) > len(component):
                self.totalcross += 1
                crossFlag = True
            collapsedEdges.append([edge, len(followers), crossFlag])
            pastNodes.extend(followers)

        collapsedEdge = sorted(collapsedEdges, key=lambda x: x[1], reverse=True)[0]
        if collapsedEdge[2] is True:
            self.crossNum += 1
        return collapsedEdge[0]

    """使用一般策略对最大k-core进行坍塌攻击,即每次删除coronaCore的一批连边"""
    def collapseCore(self, graph, kmax):
        g, collapsedEdges = graph.copy(), []
        coronaCore = nx.connected_components(nx.k_corona(G=g, k=kmax))
        count = 0
        for component in coronaCore:
            node = random.sample(list(component), 1)[0]
            edge = random.sample(list(g.edges(node)), 1)
            collapsedEdges.extend(edge)
            count += 1
        print(count)
        return collapsedEdges

    # def collapseCore(self, graph, kmax):
    #     g = graph.copy()
    #     coronaCore = nx.k_corona(G=g, k=kmax)
    #     collapsedEdges = list(coronaCore.edges).copy()
    #     # components = nx.connected_components(coronaCore)
    #     for edge in collapsedEdges:
    #         if len(collapsedEdges) <= 1:
    #             return collapsedEdges
    #         else:
    #             _, followers = self.collapsedNodes(g, edge, kmax)
    #             collapsedEdges = list(set(collapsedEdges) - set(followers))
    #             collapsedEdges.extend(edge)
    #     print(collapsedEdges)
    #     return collapsedEdges

    """基于连边两端节点的度值进行删除,选择度值最大的连边进行删除"""
    def degreeCollapse(self, graph):
        g = graph.copy()
        degree, maxWeight = dict(g.degree), 0
        for edge in g.edges:
            weight = degree[edge[0]] + degree[edge[1]]
            if weight > maxWeight:
                maxWeight = weight
                self.stack.clear()
                self.stack.push(edge)
            elif weight == maxWeight:
                self.stack.push(edge)
        collapsedEdges = self.stack.clear()
        return [collapsedEdges[0]]

    """随机删除核内的连边"""
    def randomCollapse(self, graph):
        g = graph.copy()
        collapdedEdge = random.sample(list(g.edges), 1)
        return collapdedEdge

    """CoreAttack总接口,通过strategy设置使用的攻击策略"""
    def coreAttack(self, strategy="COREATTACK"):
        self.flag = True
        g = self.G.copy()
        core = self.core.copy()
        notEmpty, deletedEdgeNum = True, 0
        while notEmpty:
            if strategy == "GreedyCOREATTACK":
                collapsedEdges = self.greedyCollapseCore(core, self.kmax)
            elif strategy == "HDE":
                collapsedEdges = self.degreeCollapse(core)
            elif strategy == "RED":
                collapsedEdges = self.randomCollapse(core)
            else:
                collapsedEdges = self.collapseCore(core, self.kmax)
            g.remove_edges_from(collapsedEdges)
            size = len(core.nodes())
            core.remove_edges_from(collapsedEdges)
            core = nx.k_core(core, self.kmax)
            notEmpty = True if core else False
            deletedEdgeNum += len(collapsedEdges)
            # print(collapsedEdges, size - len(core.nodes()))
        return [deletedEdgeNum, g]

    def AttackEpisode(self, strategy=None):
        edgeNum, nodeNum, k_edgeNum, k_nodeNum = len(self.G.edges), len(self.G.nodes), \
                                                 len(self.core.edges), len(self.core.nodes)
        ticks = time.process_time()
        results = self.coreAttack(strategy)
        ticks = time.process_time() - ticks
        _kshell = self.kshellDecomposition(results[1])
        ASR = 1 - self.attackAccuracy(self.kshell, _kshell)
        spillRate = ASR - k_nodeNum / nodeNum
        FAR = spillRate * nodeNum / (nodeNum - k_nodeNum)

        return np.array([nodeNum, edgeNum, self.kmax, k_nodeNum, k_edgeNum,
                         results[0], results[0] / edgeNum * 100,
                         ASR * 100, FAR * 100, ticks]), results[1]


def coreAttack(strategy=None):
    dirs = os.listdir(data_path)
    dirs.sort(key=lambda x: int(x[0: 2]))
    df = pd.DataFrame(columns=['|V|', '|E|', 'kmax', 'knode', 'klink', 'NDE', 'ECR%', 'ASR%', 'FAR%', 'Time'],
                      index=[i.split('.')[0] for i in dirs])
    for fullname in dirs:
        filename = fullname.split(".")[0]
        if fullname[:1] != "1":
            continue
        file = open(os.path.join(data_path, fullname), "rb")
        g = CoreAttack(nx.Graph(pkl.load(file)))

        results = np.zeros(10)
        turns = 10 if strategy == "random" else 1
        print(filename)
        for _ in range(turns):
            result = g.AttackEpisode(strategy)
            results = np.add(results, result[0])
            # print(result[1])
            # nx.write_graphml(result[1], f"./attacked_graph/{filename}_{strategy}.graphml")

        [nodesNum, edgesNum, kmax, k_nodesNum, k_edgesNum, deletedNum, ECR, ASR, FAR, ticks] = results / turns
        print("%s: %s: %.4fs, %d, %d" % (strategy, filename, ticks, deletedNum, ECR))
        df.loc[filename] = [nodesNum, edgesNum, kmax,
                            k_nodesNum, k_edgesNum,
                            deletedNum, round(ECR, 4),
                            round(ASR, 4), round(FAR, 4),
                            round(ticks, 4)]
        file.close()
        # print(f"{filename}: clustering: {nx.average_clustering(g.G):.6f}, assortatibity: {nx.degree_assortativity_coefficient(g.G):.6f}, density: {nx.density(g.G):.6f}")


        # print(df.loc[filename])
    # df.to_excel('./cache/excel/coreAttack_' + str(strategy) + '.xlsx')
    # print(df)


if __name__ == "__main__":
    p1 = mlp.Process(target=coreAttack, args=())
    p2 = mlp.Process(target=coreAttack, args=("RED",))
    p3 = mlp.Process(target=coreAttack, args=("GreedyCOREATTACK",))
    p4 = mlp.Process(target=coreAttack, args=("HDE",))
    # p1.start()
    # p2.start()
    p3.start()
    # p4.start()
    pass
