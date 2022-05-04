from __header__ import *


class Stack:
    def __init__(self):
        self.stack = []
        self.depth = 0

    def pop(self):
        if self.depth > 0:
            self.depth -= 1
            return self.stack.pop(-1)
        else:
            raise ValueError("No content in the stack")

    def push(self, item):
        self.stack.append(item)
        self.depth += 1

    def is_empty(self):
        return False if self.depth else True

    def clear(self):
        temp = self.stack.copy()
        self.stack = []
        self.depth = 0
        return temp

    def get_item(self, index):
        return self.stack[index] if index >= 0 and index < self.depth else None


class BasicMethods:
    def __init__(self, type: str):
        self.stack = Stack()
        self.type = type

    # 该函数为k-shell分解算法
    @classmethod
    def kshellDecomposition(self, edges):
        G = nx.Graph(edges)  # 用边集构建图
        kshell, k = dict(G.degree), min(dict(G.degree).values())
        while True:
            while min(dict(G.degree).values()) <= k:
                for node in list(G.degree):
                    if node[1] <= k:
                        kshell[node[0]] = k
                        G.remove_node(node[0])
                if not G:
                    return kshell
            k = min(dict(G.degree).values())

    @classmethod
    def linkShell(self, G, kshell):
        return {edge: min([kshell[edge[0]], kshell[edge[1]]]) for edge in G.edges}

    # k-shell的准确率, k原始kshell分布，k_改变的kshell分布
    @classmethod
    def attackAccuracy(self, k, k_):
        acc_absolute = [1 if k.get(key, []) == k_.get(key, []) else 0 for key in k.keys()]
        acc_attack = sum(acc_absolute) / len(k)
        return acc_attack

    @classmethod
    def get_neighbors(self, G, node, depth=1):
        output = {}
        layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
        nodes = [node]
        for i in range(1, depth + 1):
            output[i] = []
            for x in nodes:
                output[i].extend(layers.get(x, []))
            nodes = output[i]
        return output

    @classmethod
    def color_map(self, graph, distribution):
        color_map = ["Green", "Orange", "Red", "BLUE", "Violet"]
        colormap = [color_map[distribution[node] - 1] for node in graph.nodes]
        return colormap


class generateGraph(BasicMethods):
    def __init__(self, g):
        super(generateGraph, self).__init__("CoreAttack")
        self.G = nx.Graph(g)
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        self.nodes = list(self.G.nodes)
        self.edges = list(self.G.edges)
        self.kshell = self.kshellDecomposition(self.G)
        self.kMax = max(self.kshell.values())
        self.core = nx.k_core(self.G, self.kMax)

    def collapsedNodes(self, graph, deletedEdges, kMax):
        g = graph.copy()
        core = nx.k_core(g, kMax)
        coreNodes = set(core.nodes())
        g.remove_edge(*deletedEdges)
        _core = nx.k_core(g, kMax)
        _coreNodes = set(_core.nodes())
        collapsedNodes = list(coreNodes - _coreNodes)
        return collapsedNodes

    def collapse(self, graph, kMax):
        g = graph.copy()
        coronaCore = nx.k_corona(G=g, k=kMax)
        coronaNodes = sorted(nx.connected_components(coronaCore), key=len, reverse=True)
        followerIndex, combines = [], []
        while True:
            if len(coronaNodes[0]) == 1:
                for component in coronaNodes:
                    seedNode = list(g.edges(list(component)))[0]  # 返回连接的图
                    combines.append([seedNode, 1])
                return combines
            seedNode = list(coronaNodes[0])[0]
            seedEdge = list(g.edges(seedNode))[0]
            collapsedNodes = self.collapsedNodes(g, seedEdge, kMax)
            combines.append([seedEdge, len(collapsedNodes)])
            for i in range(len(coronaNodes)):
                if set(collapsedNodes) & set(coronaNodes[i]):
                    followerIndex.append(i)
            for index in sorted(followerIndex, reverse=True):
                del coronaNodes[index]
            if not coronaNodes:
                break
            coronaNodes.sort(key=lambda x: len(x), reverse=True)
            followerIndex = []
        combines.sort(key=lambda x: x[1], reverse=True)
        return combines

    def calQ(self, graph):
        Q = 0
        core = nx.k_core(graph, self.kMax)
        for edge in graph.edges:
            if edge[0] not in core.nodes:
                Q += 1
            if edge[1] not in core.nodes:
                Q += 1
        return Q / (2 * len(graph.edges))

    def coreAttackOnce(self, greedy=False, enableQ=False):
        g = self.G.copy()
        notEmpty, n = True, 0
        core = nx.k_core(g, self.kMax)
        Q = [0, self.calQ(g)]
        while notEmpty:
            edgeCombines = self.collapse(core, self.kMax)
            if greedy == True:
                g.remove_edge(*(edgeCombines[0][0]))
                core.remove_edge(*(edgeCombines[0][0]))
                n += 1
                Q.append(n / len(self.edges))
                Q.append(self.calQ(g))
            else:
                for i in range(len(edgeCombines)):
                    g.remove_edge(*(edgeCombines[i][0]))
                    core.remove_edge(*(edgeCombines[i][0]))
                    n += 1
                    Q.append(n / len(self.edges))
                    Q.append(self.calQ(g))
            core = nx.k_core(core, self.kMax)
            notEmpty = True if len(core.nodes) else False
        return n, g, Q

    def randomAttack(self):
        g = self.G.copy()
        notEmpty, n = True, 0
        core = nx.k_core(g, self.kMax)
        Q = [0, self.calQ(g)]
        while notEmpty:
            edge = random.sample(list(core.edges), 1)
            g.remove_edges_from(edge)
            core.remove_edges_from(edge)
            n += 1
            Q.append(n / len(self.edges))
            Q.append(self.calQ(g))
            core = nx.k_core(core, self.kMax)
            notEmpty = True if len(core.nodes) else False
        return n, g, Q

    def weightedAttack(self):
        g = self.G.copy()
        notEmpty, n = True, 0
        core = nx.k_core(g, self.kMax)
        Q = [0, self.calQ(g)]
        while notEmpty:
            degree = dict(core.degree)
            weights = {}
            for edge in list(core.edges):
                weights[edge] = degree[edge[0]] + degree[edge[1]]
            weights = dict(sorted(weights, key=lambda x: x[1], reverse=True))
            g.remove_edges_from(list(weights.keys())[0])
            core.remove_edges_from(list(weights.keys())[0])
            n += 1
            Q.append(n / len(self.edges))
            Q.append(self.calQ(g))
            core = nx.k_core(core, self.kMax)
            notEmpty = True if len(core.nodes) else False
        return n, g, Q



def coreAttack(greedy=True):
    path = './datasets'
    dirs = os.listdir(path)
    df = pd.DataFrame(columns=['|V|', '|E|', 'knode', 'klink', 'NDE', 'ECR', 'ASR', 'FAR', 'Time'],
                      index=[i.split('.')[0] for i in dirs])
    Qdict = {}
    for filename in dirs:
        datasets_name = filename.split('.')[0]
        print(datasets_name)
        g = generateGraph(nx.read_graphml(os.path.join(path, filename)))
        core = nx.k_core(g.G, k=g.kMax)

        ticks = time.process_time()  # The unit is seconds
        deletedEdgeNum, attackedGraph, Q = g.coreAttackOnce(greedy)
        ticks = time.process_time() - ticks
        print('Time:', ticks)
        print('NDE:', deletedEdgeNum)

        _kshell = g.kshellDecomposition(attackedGraph)

        # 基本参数
        nodesNum, edgesNum = len(g.nodes), len(g.edges)
        k_node = nx.number_of_nodes(core)
        k_edges = nx.number_of_edges(core)
        ASR = 1 - g.attackAccuracy(g.kshell, _kshell)
        # print(ASR, k_node / nodesNum)
        spillRate = ASR - k_node / nodesNum
        FAR = spillRate * nodesNum / (nodesNum - k_node)
        # print(FAR)

        df.loc[datasets_name] = [nodesNum, edgesNum,
                                 k_node, k_edges,
                                 deletedEdgeNum, round(deletedEdgeNum / edgesNum * 100, 4),
                                 round(ASR * 100, 4), round(FAR * 100, 4),
                                 round(ticks, 4)]
        Qdict[datasets_name] = Q
    # path = open("./Q_core" + str(greedy) + ".pkl", "wb")
    pkl.dump(Qdict, open("./cache/Q_core" + str(greedy) + ".pkl", "wb"))
    df.to_excel('./cache/coreAttack_' + str(greedy) + '.xlsx')
    print(df)
    pass


def randomAttack():
    path = './datasets'
    dirs = os.listdir(path)
    df = pd.DataFrame(columns=['|V|', '|E|', 'knode', 'klink', 'NDE', 'ECR', 'ASR', 'FAR', 'Time'],
                      index=[i.split('.')[0] for i in dirs])
    Qdict = {}
    for filename in dirs:
        datasets_name = filename.split('.')[0]
        # ["4GitHub", '12Brightkite', '5Gowalla', '11Autonomous']
        if datasets_name in ["4GitHub"]:
            pass
        else:
            continue
        print(datasets_name)
        g = generateGraph(nx.read_graphml(os.path.join(path, filename)))
        core = nx.k_core(g.G, k=g.kMax)

        ticks = time.process_time()  # The unit is seconds

        deletedEdgeNum, attackedGraph = 0, 0
        max, ASR, Q = 0, 0, []
        for _ in range(1):
            num, graph, Q = g.randomAttack()
            deletedEdgeNum += num
            # ASR += (1 - g.attackAccuracy(g.kshell, g.kshellDecomposition(graph))) / 10
            # Q.append(q)
        # while deletedEdgeNum != edgeNum[datasets_name]:
        #     deletedEdgeNum, attackedGraph, Q = g.randomAttack()
        #     print("repeat")
        ticks = time.process_time() - ticks
        print('Time:', ticks)
        print('NDE:', deletedEdgeNum)

        # 基本参数
        nodesNum, edgesNum = len(g.nodes), len(g.edges)
        k_node = nx.number_of_nodes(core)
        k_edges = nx.number_of_edges(core)

        # ASR = 1 - g.attackAccuracy(g.kshell, g.kshellDecomposition(attackedGraph))

        spillRate = ASR - k_node / nodesNum
        FAR = spillRate * nodesNum / (nodesNum - k_node)
        # print(FAR)

        df.loc[datasets_name] = [nodesNum, edgesNum,
                                 k_node, k_edges,
                                 round(deletedEdgeNum, 4), round(deletedEdgeNum / edgesNum * 100, 4),
                                 round(ASR * 100, 4), round(FAR * 100, 4),
                                 round(ticks, 4)]
        Qdict[datasets_name] = Q
    df.to_excel('./cache/random_1.xlsx')
    pkl.dump(Qdict, open("./cache/RED_1.pkl", "wb"))
    print(df)


def weightedAttack():
    path = './datasets'
    dirs = os.listdir(path)
    df = pd.DataFrame(columns=['|V|', '|E|', 'knode', 'klink', 'NDE', 'ECR', 'ASR', 'FAR', 'Time'],
                      index=[i.split('.')[0] for i in dirs])
    Qdict = {}
    for filename in dirs:
        datasets_name = filename.split('.')[0]
        print(datasets_name)
        g = generateGraph(nx.read_graphml(os.path.join(path, filename)))
        core = nx.k_core(g.G, k=g.kMax)

        ticks = time.process_time()  # The unit is seconds
        deletedEdgeNum, attackedGraph, Q = g.randomAttack()
        ticks = time.process_time() - ticks
        print('Time:', ticks)
        print('NDE:', deletedEdgeNum)

        _kshell = g.kshellDecomposition(attackedGraph)

        # 基本参数
        nodesNum, edgesNum = len(g.nodes), len(g.edges)
        k_node = nx.number_of_nodes(core)
        k_edges = nx.number_of_edges(core)
        ASR = 1 - g.attackAccuracy(g.kshell, _kshell)
        # print(ASR, k_node / nodesNum)
        spillRate = ASR - k_node / nodesNum
        FAR = spillRate * nodesNum / (nodesNum - k_node)
        # print(FAR)

        df.loc[datasets_name] = [nodesNum, edgesNum,
                                 k_node, k_edges,
                                 deletedEdgeNum, round(deletedEdgeNum / edgesNum * 100, 4),
                                 round(ASR * 100, 4), round(FAR * 100, 4),
                                 round(ticks, 4)]
        Qdict[datasets_name] = Q
    df.to_excel('./cache/weighted.xlsx')
    pkl.dump(Qdict, open("./cache/HDE.pkl", "wb"))
    print(df)
    pass


if __name__ == "__main__":
    # weightedAttack()
    # coreAttack(False)
    # coreAttack(True)
    randomAttack()
