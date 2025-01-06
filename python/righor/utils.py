import re
import sys
from pathlib import Path
from righor import _righor
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import mutual_info_score
import os


def fix_number_threads(num_threads):
    os.environ["RAYON_NUM_THREADS"] = f"{num_threads}"


def rc(seq):
    """ Return the reverse-complement of a nucleotide sequences """
    dct = {"A": "T", "G": "C", "T": "A", "C": "G", "R": "Y", "Y": "R", "S": "S",
           "W": "W", "K": "M", "M": "K", "B": "V", "D": "H", "H": "D", "V": "B", "N": "N"}
    return "".join([dct[a] for a in seq][::-1])


_bases = "TCAG"
_codons = [a + b + c for a in _bases for b in _bases for c in _bases]
_aminoacids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
_to_aminoacid = dict(zip(_codons, _aminoacids))
_to_codons = {}
for a, c in zip(_aminoacids, _codons):
    if a in _to_codons:
        _to_codons[a].append(c)
    else:
        _to_codons[a] = [c]


def n_to_aa(x):
    """ Transform the sequence x to its nucleotide form """
    return "".join([_to_aminoacid[x[i:i+3]] for i in range(0, len(x), 3)])


def aa_to_n(x):
    """ Return all the possible versions of the aa sequence x
    in nucleotide form """
    for s in itertools.product(*(_to_codons[a] for a in x)):
        yield "".join(s)

def cut_mod3(x: str):
    """ Cut the end of a str so that its length is proportional to 3 """
    return x[:3*(len(x)//3)]

def load_model(species: str, chain: str, identifier=None):
    """ Load the appropriate model, for example:
    load_model("human", "trb", id="emerson") will load the human trb model based
    on emerson data.
    load_model("human", "tra") will load the default trb model
    """
    #    model_dir = .absolute().as_posix()
    # very ugly but maturin seems to force that
    venv_root = Path(sys.prefix)
    path_model = venv_root / Path("righor_models")
    if not path_model.exists():  # let's guess (ugly)
        righor_path = Path(_righor.__file__).as_posix()
        venv_root = Path(righor_path.split('/lib/python')[0])
        path_model = venv_root / Path("righor_models")
    if not path_model.exists():  # local mode
        path_model = (Path(_righor.__file__).parent.parent.parent /
                      Path("righor.data") / Path("data") / Path("righor_models"))
    if not path_model.exists():
        raise RuntimeError(
            "Error with the installation. Data files not found.")
    model_dir = path_model.absolute().as_posix()

    try:
        model = _righor.Model.load_model(species,
                                         chain,
                                         model_dir,
                                         identifier)
    except:
        if identifier is None:
            raise(ValueError(
                f"Wrong species ({species}) and/or chain ({chain})"))
        else:
            raise(ValueError(
                f"Wrong species ({species}) and/or chain ({chain}) and/or id ({id})"))
    return model

def load_model_from_files(json=None, path_params=None,
                         path_marginals=None,
                         path_anchor_vgene=None,
                         path_anchor_jgene=None):
    """ Load a model from a file.
    If json is not None, load the model from the json file.
    If all the path_* parameters are not None, load from a `Igor Format`
    """
    if json is not None:
        try:
            return _righor.Model.load_json(json)
        except:
            raise(ValueError("Invalid json file, or invalid filename."))
    if path_params is not None and path_marginals is not None \
       and path_anchor_vgene is not None and path_anchor_jgene is not None:
        try:
            return _righor.Model.load_from_files(path_params, path_marginals,
                                      path_anchor_vgene, path_anchor_jgene)
        except:
            raise(ValueError("Invalid igor model files, or wrong filenames."))

    else:
        raise(ValueError(
        f"Absent files"))



def genes_matching(x: str, model):
    """ Map relatively standard gene name to
        the genes used in Igor/Righor.
        In general return a bit more than needed if there's a doubt
        So TRAV1-1*13 will return all TRAV1-1s, but TRAV1-1*1 will return TRAV1-01*01 and
        TRAV1542 will return all TRAV.
        It's far from perfect.
        @ Arguments:
        * x: V or J gene name, form: TYPE##-##*##, or TYPE##-##
        or TYPE[V,J]##, where ## can be interpreted as digits/letters
        and TYPE = "IGH","IGK","IGL" or "TRB"/"TRA"/"TRG"/"TRD"
        * model: righor.Model object.
        @ Return:
        * list of righor Gene object (x.name to get their names)
        @ Example:
        "IGHV07" -> ["IGHV7-34-1*01", "IGHV7-34-1*02", "IGHV7-4-1*01",
                     "IGHV7-4-1*02", "IGHV7-4-1*03","IGHV7-4-1*04",
                     "IGHV7-4-1*05", "IGHV7-40*01", "IGHV7-81*01"]
        "TRBV07-002*4 -> ["TRBV7-2*04"]
    """

    regex = (r"^(TC?RB|TC?RA|IGH|IGK|IGL|TC?RG|TC?RD)(?:\w+)?(V|D|J)"
             r"([\w-]+)?(?:/DV\d+)?(?:\*(\d+))?(?:/OR.*)?$")
    g = re.search(regex, x)

    chain = None
    gene_type = None
    gene_id = None
    allele = None

    if g is None:
        raise ValueError("Gene {} does not have a valid name".format(x))
    chain = g.group(1)
    gene_type = g.group(2)
    gene_id = g.group(3)
    allele = None if g.group(4) is None else int(g.group(4))

    if chain is None or gene_type is None:
        raise ValueError("Gene {} does not have a valid name".format(x))

    # check if gene_id contain something of the form
    # ##-## where ## is a digit or ##S##
    gene_id_1 = None
    gene_id_2 = None
    if gene_id is not None:
        g = re.search(r'(\d+)(?:[-S](\d+))?', gene_id)
        if g is not None:
            if g.span()[1] >= 3 and g.group(2) is not None:
                gene_id_1 = int(g.group(1))
                gene_id_2 = int(g.group(2))
            else:
                gene_id_1 = int(g.group(1))

    chain = chain.replace('TCR', 'TR')  # in case there's a C

    possible_genes = igor_genes(chain, gene_type, model)

    if allele is not None and gene_id_1 is not None and gene_id_2 is not None:
        guess = [a[-1] for a in possible_genes if a[1] == gene_id_1
                 and a[2] == gene_id_2 and a[4] == allele]
        if guess != []:
            return guess
    if gene_id_1 is not None and gene_id_2 is not None:
        guess = [a[-1] for a in possible_genes if a[1] == gene_id_1
                 and a[2] == gene_id_2]
        if guess != []:
            return guess
    if allele is not None and gene_id_1 is not None:
        guess = [a[-1] for a in possible_genes if a[1] == gene_id_1
                 and a[4] == allele]
        if guess != []:
            return guess
    if gene_id_1 is not None:
        guess = [a[-1] for a in possible_genes if a[1] == gene_id_1]
        if guess != []:
            return guess

    # if everything else failed return all
    return [a[-1] for a in possible_genes]


def igor_genes(chain: str, gene_type: str, model):
    """ Read the model and return all the genes matching the chain and gene_type.
        chain: TR/IG
        gene_type: V/J
        It returns the full gene name, plus the gene family, its name, its allele, plus the Gene object
    """
    regex = (r"(\d+)(?:P)?(?:[\-S](\d+)(?:D)?(?:\-(\d+))?)?"
             r"(?:/DV\d+)?(?:-NL1)?(?:\*(\d+))?")  # match all the IGoR gene names

    lst = []

    list_genes = None
    if gene_type == "V":
        list_genes = model.v_segments
    elif gene_type == "J":
        list_genes = model.j_segments
    else:
        raise ValueError("Gene type {} is not valid".format(gene_type))

    key = chain + gene_type
    for gene_obj in list_genes:
        gene = gene_obj.name
        try:
            lst.append(
                tuple(
                    [gene] + [None if a is None else int(a) for a in
                              re.search(key + regex, gene).groups()]
                    + [gene_obj]
                ))
        except AttributeError:
            raise ValueError(
                f"{key} does not match. Check if the gene name and the model are compatible (e.g. TRA for a TRB/IGL model)") from None

    return lst


def plot_vdj(*args, plots_kws=None):
    """ Plot all the marginals of one or more VDJ models.
        The order of V/D/J genes is based on the first model.
        Arguments:
        - *args: VDJ model objects
        - plots_kws: list of plot options for each model, in dict form
        for ex: [{'label': 'my model', 'color': '#131313', ls: '-'}, ...]
        Usage:
        plot_vdj(model1, model2, model3, [{'label'='m1', 'color'='r'}, ('label'='m2', 'color'='b'}, {'label'='m3', 'color'='g'}])
    """

    fig = plt.figure(constrained_layout=True, figsize=(16, 18))
    gs = fig.add_gridspec(6, 2)
    ax_V = fig.add_subplot(gs[0, :])
    ax_J = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    ax_delV = fig.add_subplot(gs[2, 0])
    ax_delJ = fig.add_subplot(gs[2, 1])
    ax_delD5 = fig.add_subplot(gs[3, 0])
    ax_delD3 = fig.add_subplot(gs[3, 1])
    ax_insVD = fig.add_subplot(gs[4, 0])
    ax_insDJ = fig.add_subplot(gs[4, 1])
    ax_nucVD = fig.add_subplot(gs[5, 0])
    ax_nucDJ = fig.add_subplot(gs[5, 1])

    ax_V.set_xlabel('V genes')
    ax_J.set_xlabel('J genes')
    ax_D.set_xlabel('D genes')

    for ax in [ax_V, ax_D, ax_J]:
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelrotation=90)

    ax_delV.set_xlabel('# V deletions')
    ax_delJ.set_xlabel('# J deletions')
    ax_delD3.set_xlabel('# D3 deletions')
    ax_delD5.set_xlabel('# D5 deletions')
    ax_insVD.set_xlabel('# insertions between V and D')
    ax_insDJ.set_xlabel('# insertions between D and J')
    ax_nucVD.set_xlabel('nuc. transitions at VD')
    ax_nucVD.tick_params(axis='x', labelrotation=90)
    ax_nucDJ.set_xlabel('nuc. transitions at DJ')
    ax_nucDJ.tick_params(axis='x', labelrotation=90)

    # Group all the v_genes used in the model
    # ignore the alleles
    all_v_genes = list(set([g.name.split('*')[0]
                            for m in args for g in m.v_segments]))
    all_d_genes = list(set([g.name.split('*')[0]
                            for m in args for g in m.d_segments]))
    all_j_genes = list(set([g.name.split('*')[0]
                            for m in args for g in m.j_segments]))
    order_vs = range(len(all_v_genes))
    order_ds = range(len(all_v_genes))
    order_js = range(len(all_j_genes))

    if plots_kws is None:
        plots_kws = [{} for m in args]

    for idx_model, (m, opt) in enumerate(zip(args, plots_kws)):
        # for each V gene, compute P(V) for the V gene (ignoring alleles)
        proba_vs = m.p_vdj.sum(axis=(1, 2))
        p_v_sans_alleles = np.zeros(len(all_v_genes))
        for (v, pv) in zip(m.v_segments, proba_vs):
            idx = [ii for (ii, vsansallele) in enumerate(
                all_v_genes) if vsansallele == v.name.split('*')[0]][0]
            p_v_sans_alleles[idx] += pv

        # fix the order based on the first element
        if idx_model == 0:
            order_vs = np.argsort(p_v_sans_alleles)[::-1]

        ax_V.plot(np.array(all_v_genes)[order_vs],
                  p_v_sans_alleles[order_vs], **opt)

        # same for J
        proba_js = m.p_vdj.sum(axis=(0, 1))
        p_j_sans_alleles = np.zeros(len(all_j_genes))
        for (j, pj) in zip(m.j_segments, proba_js):
            idx = [ii for (ii, jsansallele) in enumerate(
                all_j_genes) if jsansallele == j.name.split('*')[0]][0]
            p_j_sans_alleles[idx] += pj

        # fix the order based on the first element
        if idx_model == 0:
            order_js = np.argsort(p_j_sans_alleles)[::-1]

        ax_J.plot(np.array(all_j_genes)[order_js],
                  p_j_sans_alleles[order_js], **opt)

        # same for D
        proba_ds = m.p_vdj.sum(axis=(0, 2))
        p_d_sans_alleles = np.zeros(len(all_d_genes))
        for (d, pd) in zip(m.d_segments, proba_ds):
            idx = [ii for (ii, dsansallele) in enumerate(
                all_d_genes) if dsansallele == d.name.split('*')[0]][0]
            p_d_sans_alleles[idx] += pd

        # fix the order based on the first element
        if idx_model == 0:
            order_ds = np.argsort(p_d_sans_alleles)[::-1]

        ax_D.plot(np.array(all_d_genes)[order_ds],
                  p_d_sans_alleles[order_ds], **opt)

        # Now the deletions
        ax_delV.plot(
            range(m.range_del_v[0], m.range_del_v[1]+1),
            m.p_del_v_given_v@m.p_vdj.sum(axis=(1, 2)), **opt
        )

        ax_delJ.plot(
            range(m.range_del_j[0], m.range_del_j[1]+1),
            m.p_del_j_given_j@m.p_vdj.sum(axis=(0, 1)), **opt
        )

        ax_delD5.plot(
            range(m.range_del_d5[0], m.range_del_d5[1]+1),
            (m.p_del_d5_del_d3.sum(axis=(1))@m.p_vdj.sum(axis=(0, 2))), **opt
        )

        ax_delD3.plot(
            range(m.range_del_d3[0], m.range_del_d3[1]+1),
            (m.p_del_d5_del_d3.sum(axis=(0))@m.p_vdj.sum(axis=(0, 2))), **opt
        )

        # And the insertions
        ax_insVD.plot(
            range(0, m.p_ins_vd.shape[0]),
            m.p_ins_vd, **opt
        )
        ax_insDJ.plot(
            range(0, m.p_ins_dj.shape[0]),
            m.p_ins_dj, **opt
        )

        ax_nucVD.scatter(
            [f'{a}→{b}' for a in 'ACGT' for b in 'ACGT'],
            m.markov_coefficients_vd.flatten(), **opt
        )

        ax_nucDJ.scatter(
            [f'{a}→{b}' for a in 'ACGT' for b in 'ACGT'],
            m.markov_coefficients_dj.flatten(), **opt
        )

    for ax in [ax_V, ax_D, ax_J]:
        ylim = ax.get_ylim()
        ylim = (max(ylim[0], 1e-6), 1.)
        ax.set_ylim(ylim)

    if 'label' in plots_kws[0]:
        for ax in fig.axes:
            ax.legend()

    return fig


def plot_vj(*args, plots_kws=None):
    """ Plot all the marginals of one or more VJ models.
        The order of V/J genes is based on the first model.
        Arguments:
        - *args: VJ model objects
        - plots_kws: list of plot options for each model, in dict form
        for ex: [{'label': 'my model', 'color': '#131313', ls: '-'}, ...]
        Usage:
        plot_vdj(model1, model2, model3, [{'label'='m1', 'color'='r'}, ('label'='m2', 'color'='b'}, {'label'='m3', 'color'='g'}])
    """

    fig = plt.figure(constrained_layout=True, figsize=(16, 12))
    gs = fig.add_gridspec(4, 2)
    ax_V = fig.add_subplot(gs[0, :])
    ax_J = fig.add_subplot(gs[1, :])
    ax_delV = fig.add_subplot(gs[2, 0])
    ax_delJ = fig.add_subplot(gs[2, 1])
    ax_insVJ = fig.add_subplot(gs[3, 0])
    ax_nucVJ = fig.add_subplot(gs[3, 1])

    ax_V.set_xlabel('V genes')
    ax_J.set_xlabel('J genes')

    for ax in [ax_V, ax_J]:
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelrotation=90)

    ax_delV.set_xlabel('# V deletions')
    ax_delJ.set_xlabel('# J deletions')
    ax_insVJ.set_xlabel('# insertions between V and J')
    ax_nucVJ.set_xlabel('nuc. transitions at VJ')
    ax_nucVJ.tick_params(axis='x', labelrotation=90)

    # Group all the v_genes used in the model
    # ignore the alleles
    all_v_genes = list(set([g.name.split('*')[0]
                            for m in args for g in m.v_segments]))
    all_j_genes = list(set([g.name.split('*')[0]
                            for m in args for g in m.j_segments]))
    order_vs = range(len(all_v_genes))
    order_js = range(len(all_j_genes))

    if plots_kws is None or len(plots_kws) != len(args):
        plots_kws = [{} for m in args]

    for idx_model, (m, opt) in enumerate(zip(args, plots_kws)):
        # for each V gene, compute P(V) for the V gene (ignoring alleles)
        proba_vs = m.p_vj.sum(axis=(1))
        p_v_sans_alleles = np.zeros(len(all_v_genes))
        for (v, pv) in zip(m.v_segments, proba_vs):
            idx = [ii for (ii, vsansallele) in enumerate(
                all_v_genes) if vsansallele == v.name.split('*')[0]][0]
            p_v_sans_alleles[idx] += pv

        # fix the order based on the first element
        if idx_model == 0:
            order_vs = np.argsort(p_v_sans_alleles)[::-1]

        ax_V.plot(np.array(all_v_genes)[order_vs],
                  p_v_sans_alleles[order_vs], **opt)

        # same for J
        proba_js = m.p_vj.sum(axis=(0))
        p_j_sans_alleles = np.zeros(len(all_j_genes))
        for (j, pj) in zip(m.j_segments, proba_js):
            idx = [ii for (ii, jsansallele) in enumerate(
                all_j_genes) if jsansallele == j.name.split('*')[0]][0]
            p_j_sans_alleles[idx] += pj

        # fix the order based on the first element
        if idx_model == 0:
            order_js = np.argsort(p_j_sans_alleles)[::-1]

        ax_J.plot(np.array(all_j_genes)[order_js],
                  p_j_sans_alleles[order_js], **opt)

        # Now the deletions
        ax_delV.plot(
            range(m.range_del_v[0], m.range_del_v[1]+1),
            m.p_del_v_given_v@m.p_vj.sum(axis=(1)), **opt
        )

        ax_delJ.plot(
            range(m.range_del_j[0], m.range_del_j[1]+1),
            m.p_del_j_given_j@m.p_vj.sum(axis=(0)), **opt
        )

        # And the insertions
        ax_insVJ.plot(
            range(0, m.p_ins_vj.shape[0]),
            m.p_ins_vj, **opt
        )

        ax_nucVJ.scatter(
            [f'{a}→{b}' for a in 'ACGT' for b in 'ACGT'],
            m.markov_coefficients_vj.flatten(), **opt
        )

    for ax in [ax_V, ax_J]:
        ylim = ax.get_ylim()
        ylim = (max(ylim[0], 1e-6), ylim[1])
        ax.set_ylim(ylim)

    if 'label' in plots_kws[0]:
        for ax in fig.axes:
            ax.legend()

    return fig



def stationary_distribution(markov_coefficients):
    evals, evecs = np.linalg.eig(markov_coefficients.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()
    return stationary.real


# def airr_dataframe(results, model):
#     """ Return a list of sequences (either generated or evaluated) in
#     AIRR format.
#     """
#     fields = {
#         'v_call', 'd_call', 'j_call', 'sequence_inferred', 'sequence_inferred_aa', 'germline_alignment', 'germline_alignment_aa',
#         'junction', 'junction_aa', 'cdr1', 'cdr1_aa', 'cdr2', 'cdr2_aa', 'cdr3', 'cdr3_aa', 'fwr1', 'fwr1_aa', 'fwr2', 'fwr2_aa',
#         'fwr3', 'fwr3_aa', 'fwr4', 'fwr4_aa', 'junction_length', 'junction_aa_length', 'd_frame', 'd2_frame', 'productive', 'stop_codon'}

#     lst_dct = []
#     for r in results:
#         if hasattr(x, 'recombination_event'): #if generated sequences
#             dct['v_call'] = r.v_gene
#             dct['j_call'] = r.j_gene
#             dct['d_call'] = model.d_segments[r.recombination_event.d_index].name
#             dct['sequence_inferred'] = r.full_seq
#             dct['sequence_inferred_aa'] = n_to_aa(dct['sequence_inferred'])
#             dct['germline_alignment'] = True # sequence without error
#             dct['germline_alignment_aa'] = n_to_aa(dct['germline_alignment'])
#             dct['junction'] = r.junction_nt
#             dct['junction_aa'] = r.junction_aa
#             dct['cdr3_nt'] = r.junction_nt[3:-3]
#             dct['cdr3_aa'] = r.junction_aa[1:-1] if r.junction_aa is not None else None
#             dct['productive'] = (r.junction_nt % 3 == 0 &&
#                                  "*" not in r.junction_aa &&
#                                  "C" == r.junction_aa[0] &&
#                                  r.junction_aa[-1] in ['F', 'W'])
#             dct['stop_codon'] = "*" in dct['sequence_inferred_aa']
#         else:
#             dct['v_call'] = r.best_v_gene
#             dct['j_call'] = r.best_j_gene
#             dct['d_call'] = r.best_d_gene if r.best_d_gene != "Empty_D_gene'" else None
#             dct['sequence_inferred'] = r.d_sequence
#             dct['sequence_inferred'] = n_to_aa(cut_mod3(r.d_sequence))\
#                                                if len(r.best_event.junction) % 3 == 0 else None
#             dct['germline_alignment'] = r.germline_sequence
#             dct['germline_alignment_aa'] = n_to_aa(cut_mod3(r.germline_sequence))\
#                 if len(r.best_event.junction) % 3 == 0 else None

#             dct['junction'] = r.best_event.junction
#             dct['junction_aa'] = n_to_aa(r.best_event.junction) if len(r.best_event.junction) % 3 == 0 else None
#             dct['cdr3_nt'] = dct['junction'][3:-3]
#             dct['cdr3_aa'] = dct['junction_aa'][1:-1] if dct['junction_aa'] is not None else None

#         lst_dct += [dct]

#     df = pl.DataFrame(dct)
#     df = df.drop(df.columns[pl.col("*").is_null().all()])
#     return df



def get_mutual_information_model(model, variables=[], n=10000):
    """ Generate `n` sequences through the model and measure the mutual information
    between the variables in `variables`
    """
    generator = model.generator(seed=None)
    sequences = [generator.generate(False) for _ in range(n)]
    df = airr_dataframe(sequences)
    return get_mutual_information(df)


def mutual_info_score(df):
    """ Return the mutual information between all the columns of a dataframe """
    cols = df.columns
    mi_matrix = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            mi_matrix.loc[col1, col2] = mutual_info_score(df[col1], df[col2])
    return mi_matrix


def get_mutual_information(df_airr):
    """ Return the mutual information between the variables `variables`.
    """
    keys = ['v_call', 'j_call', 'd_call',
            'len_vd_insertions', 'len_dj_insertions', 'vj_insertions',
            'deletion_3_end_v', 'deletion_5_end_d',
            'deletion_3_end_d', 'deletion_5_end_j']
    keys = list(set(keys) & set(df_airr.columns))
    return mutual_info_score(df_airr[keys])


def get_entropy(model):
    """ Return the entropic contribution of each feature of the model.
    """
    def alogb(a, b):
        x = np.zeros(a.shape)
        x[a != 0] = (a[a != 0] * np.log2(b[a != 0]))
        return x

    # genes
    info_v = alogb(model.p_v, model.p_v)
    entropy_v = info_v.sum()

    p_vj = model.p_vdj.sum(axis=1)
    p_j = np.zeros(p_vj.shape)
    p_j = np.divide(p_vj, model.p_vdj.sum(axis=(1, 2))[:, None], where=(p_vj != 0))
    info_j = alogb(model.p_vdj.sum(axis=1), p_j)
    entropy_j = info_j.sum()

    p_d = np.divide(model.p_vdj, model.p_vdj.sum(axis=1)[:, None, :], where=(model.p_vdj != 0))
    info_d = alogb(model.p_vdj, p_d)
    entropy_d = info_d.sum()

    # deletion
    info_delv_v = alogb(model.p_del_v_given_v * model.p_v[None, :],  model.p_del_v_given_v)
    entropy_delv = info_delv_v.sum()

    p_delj_v_j =  model.p_del_j_given_j[:, None, :] * model.p_vdj.sum(axis=1)[None, :, :]
    info_delj_v_j = alogb(p_delj_v_j, np.broadcast_to(model.p_del_j_given_j[:, None, :], p_delj_v_j.shape))
    entropy_delj = info_delj_v_j.sum()

    p_del_d_v_d_j = \
        model.p_del_d5_del_d3[:, :, None, :, None] *\
        model.p_vdj[None, None, :, :, :]
    info_deld = alogb(p_del_d_v_d_j,
                      np.broadcast_to(model.p_del_d5_del_d3[:, :, None, :, None], p_del_d_v_d_j.shape))
    entropy_deld = info_deld.sum()

    # insertion (more complicated, need to deal with the entropy of the Markov chain)

    stat_distr = stationary_distribution(model.markov_coefficients_vd)
    entropy_markov_vd = alogb(
        stat_distr[:, None] * model.markov_coefficients_vd,
        model.markov_coefficients_vd).sum()
    entropy_ins_vd = \
        alogb(model.p_ins_vd, model.p_ins_vd).sum() +\
        sum([model.p_ins_vd[n] * n * entropy_markov_vd
             for n in range(model.p_ins_vd.shape[0])])

    stat_distr = stationary_distribution(model.markov_coefficients_dj)
    entropy_markov_dj = alogb(
        stat_distr[:, None] * model.markov_coefficients_dj,
        model.markov_coefficients_vd).sum()
    entropy_ins_dj = \
        alogb(model.p_ins_dj, model.p_ins_dj).sum() +\
        sum([model.p_ins_dj[n] * n * entropy_markov_dj
             for n in range(model.p_ins_dj.shape[0])])

    stat_distr = stationary_distribution(model.markov_coefficients_vj)
    entropy_markov_vj = alogb(
            stat_distr[:, None] * model.markov_coefficients_vj,
        model.markov_coefficients_vj).sum()
    entropy_ins_vj = \
        alogb(model.p_ins_vj, model.p_ins_vj).sum() +\
        sum([model.p_ins_vj[n] * n * entropy_markov_vj
             for n in range(model.p_ins_vj.shape[0])])


    # need to do the VJ situation too, TODO
    return {
        'entropy_v': entropy_v,
        'entropy_j': entropy_j,
        'entropy_d': entropy_d,
        'entropy_delv': entropy_delv,
        'entropy_delj': entropy_delj,
        'entropy_deld': entropy_deld,
        'entropy_ins_vd': entropy_ins_vd,
        'entropy_ins_dj': entropy_ins_dj,
    }

def plot_mutual_information(model):
    pass

def plot_entropy(model):
    pass
