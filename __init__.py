from CoreAttack import *


if __name__ == "__main__":
    dirs = os.listdir("./graphml/")
    dirs.sort(key=lambda x: int(x[0: 2]))
    for fullname in dirs:
        filename = fullname.split(".")[0]
        if fullname[:2] != "22":
            continue
        g = CoreAttack(nx.read_graphml(os.path.join("./graphml/", fullname)))
        corona = nx.k_corona(G=g.core, k=g.kmax)
        components = list(nx.connected_components(corona))
        length = 0
        for component in components:
            length += len(component)
        print(filename, len(components), length / len(g.core.nodes), len(components))
