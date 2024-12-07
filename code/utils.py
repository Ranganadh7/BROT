#!/usr/bin/env python
# coding: utf-8

# In[1]:


def unpack_results(OUTPUT_PATH):
    
    from os import listdir
    from os.path import isfile, join
    import pickle as pkl
    
    file_names = [f for f in listdir(OUTPUT_PATH) if isfile(join(OUTPUT_PATH, f))]
    
    for file in file_names:
        if "OTexec_True_GDexec_True" in file:
            if "metadata" in file:
                metadata_BROT = pkl.load(open(OUTPUT_PATH + file, "rb"))
            if "results" in file:
                results_BROT = pkl.load(open(OUTPUT_PATH + file, "rb"))
        if "OTexec_True_GDexec_False" in file:
            if "metadata" in file:
                metadata_OT = pkl.load(open(OUTPUT_PATH + file, "rb"))
            if "results" in file:
                results_OT = pkl.load(open(OUTPUT_PATH + file, "rb"))
        if "OTexec_False_GDexec_True" in file:
            if "metadata_OT" in file:
                metadata_PSGD = pkl.load(open(OUTPUT_PATH + file, "rb"))
            if "results_OT_" in file:
                results_PSGD = pkl.load(open(OUTPUT_PATH + file, "rb"))

    metadata = {"BROT": metadata_BROT, "PSGD": metadata_PSGD, "OT": metadata_OT}
    results = {"BROT": results_BROT, "PSGD": results_PSGD, "OT": results_OT}

    return metadata, results


# In[2]:


def plot_J_Omega(results):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

    # normalization coefficients
    norm_S = 0.5 # S is not normalized in input
    norm_w = np.sum(results["OT"]["wevol"]) # results in code are serialized in OT units
    
    J_BROT = results["BROT"]["Jevol"]/(norm_S*norm_w)
    J_PSGD = results["PSGD"]["Jevol"]/(norm_S*norm_w)
    J_OT = results["OT"]["Jevol"]/(norm_S*norm_w)
    
    ax[0].scatter([0,1,2], [J_BROT, J_PSGD, J_OT], color=["C2","C1","C0"], s=100)
    ax[0].grid(axis="y", lw=1, ls="dotted", color="silver")
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xticklabels(["BROT", "PSGD", "OT"])
    ax[0].tick_params(axis='both', labelsize=15)
    ax[0].set_ylabel(r"$J$", size=20, rotation=0, labelpad=20)
    
    
    O_BROT = results["BROT"]["Omegaevol"]/norm_S**2
    O_PSGD = results["PSGD"]["Omegaevol"]/norm_S**2
    O_OT = results["OT"]["Omegaevol"]/norm_S**2
    
    ax[1].scatter([0,1,2], [O_BROT, O_PSGD, O_OT], color=["C2","C1","C0"], s=100)
    ax[1].grid(axis="y", lw=1, ls="dotted", color="silver")
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xticklabels(["BROT", "PSGD", "OT"])
    ax[1].tick_params(axis='both', labelsize=15)
    ax[1].set_ylabel(r"$\Omega$", size=20, rotation=0, labelpad=20)


# In[3]:


def plot_network_F(GRAPH, commodities, F, ax, title):
    
    import networkx as nx
    from matplotlib.patches import Patch
    import numpy as np
    
    scaleF = 10
    
    pos = {node[0]: node[1]["pos"] for node in GRAPH.nodes(data=True)}
    
    nx.draw_networkx_nodes(
    GRAPH,
    pos=pos,
    nodelist=GRAPH.nodes(),
    node_size=0,
    node_color="silver",
    node_shape="o",
    ax=ax,
    edgecolors="black",
    )

    nx.draw_networkx_nodes(
    GRAPH,
    pos=pos,
    nodelist=commodities,
    node_size=np.array([1,1])*100,
    node_color=["red", "blue"],
    node_shape="o",
    ax=ax,
    linewidths=2,
    edgecolors=["red", "blue"],
    )
    
    lab = dict()
    lab[commodities[0]] = r"+"
    lab[commodities[1]] = r"─"
    nx.draw_networkx_labels(GRAPH, pos=pos, labels=lab, font_color='white', font_weight='bold',font_size=16, ax=ax)
    
    nx.draw_networkx_edges(
    GRAPH,
    pos=pos,
    width=F*scaleF,
    edge_color="black",
    style="solid",
    ax=ax,
    connectionstyle="arc3",
    )
    
    ax.set_title(title, size=20)
    
    legend_elements = [Patch(facecolor='black', edgecolor='None', lw=0.5, hatch="", label=r'$||F_e||_1$')]

    ax.legend(handles=legend_elements, loc="best",
    fancybox=False, framealpha=1.0, facecolor="None", edgecolor="None",
    fontsize=14, borderpad=0.1, labelspacing=0.05, borderaxespad=0.1,ncol=1,columnspacing=0, handlelength=2, handletextpad=0.5)

    


# In[4]:


def plot_network_w(GRAPH, commodities, rho, ax, title):
    
    import networkx as nx
    import matplotlib
    import numpy as np

    cmr_cmap = matplotlib.cm.get_cmap(name="coolwarm")
    
    pos = {node[0]: node[1]["pos"] for node in GRAPH.nodes(data=True)}
    
    nx.draw_networkx_nodes(
    GRAPH,
    pos=pos,
    nodelist=GRAPH.nodes(),
    node_size=0,
    node_color="silver",
    node_shape="o",
    ax=ax,
    edgecolors="black",
    )

    nx.draw_networkx_nodes(
    GRAPH,
    pos=pos,
    nodelist=commodities,
    node_size=np.array([1,1])*100,
    node_color=["red", "blue"],
    node_shape="o",
    ax=ax,
    linewidths=2,
    edgecolors=["red", "blue"],
    )
    
    lab = dict()
    lab[commodities[0]] = r"+"
    lab[commodities[1]] = r"─"
    nx.draw_networkx_labels(GRAPH, pos=pos, labels=lab, font_color='white', font_weight='bold',font_size=16, ax=ax)
        
    max_color = np.max(abs(rho))
    
    nx.draw_networkx_edges(
    GRAPH,
    pos=pos,
    width=5,
    edge_color=rho,
    edge_vmin=-max_color,
    edge_vmax=+max_color,
    edge_cmap=cmr_cmap,
    style="solid",
    ax=ax,
    connectionstyle="arc3",
    )
    
    ax.set_title(title, size=20)


# In[5]:


def plot_colorbar(fig, rho, ax):

    import matplotlib
    import matplotlib.ticker as mticker
    import warnings
    import numpy as np

    print("rho_X := w_X - w_OT, with X = BROT, GD")

    warnings.filterwarnings("ignore")

    max_color = np.max(abs(rho))

    ax = fig.add_axes([0.915,0.1, 0.02, 0.8])
    cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical',
                                   cmap=matplotlib.cm.get_cmap(name="coolwarm"),
                                   norm=matplotlib.colors.Normalize(-max_color, +max_color),
                                   ticks=[-max_color, 0, +max_color])

    label_format = ['{:,.3f}', '{:,.1f}', '{:,.1f}']
    ticks_loc = ax.get_yticks().tolist()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_yticklabels(["0" for n, x in enumerate(ticks_loc)])

    cb.ax.set_yticklabels([r'-max', '0', r'+max'], size=12)  # vertically oriented colorbar
    cb.ax.set_title(r'$\rho$', fontsize=12)  # vertically oriented colorbar
    cb.ax.tick_params(axis='y', which='major', pad=2)
    cb.ax.tick_params(length=1, width=0.5, which="major")

