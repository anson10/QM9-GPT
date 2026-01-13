import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import py3Dmol
from stmol import showmol

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None: self.out += self.bias
        return self.out
    def parameters(self): return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps, self.momentum, self.training = eps, momentum, False 
        self.gamma, self.beta = torch.ones(dim), torch.zeros(dim)
        self.running_mean, self.running_var = torch.zeros(dim), torch.ones(dim)
    def __call__(self, x):
        xmean, xvar = self.running_mean, self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out
    def parameters(self): return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x): return torch.tanh(x)
    def parameters(self): return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, IX): return self.weight[IX]
    def parameters(self): return [self.weight]

class FlattenConsecutive:
    def __init__(self, n): self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1: x = x.squeeze(1)
        return x
    def parameters(self): return []

class Sequential:
    def __init__(self, layers): self.layers = layers
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self): return [p for layer in self.layers for p in layer.parameters()]


st.set_page_config(page_title="QM9-GPT Discovery Lab", layout="wide")

# Custom Lab CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #4F46E5; color: white; height: 3.5em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_project_data():
    df = pd.read_csv('data/processed/qm9_clean.csv')
    smiles_list = df['smiles'].astype('str').tolist()
    training_set = set(smiles_list)
    chars = sorted(list(set(''.join(smiles_list))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos, len(itos), df['gap_ev'].min(), df['gap_ev'].max(), training_set

stoi, itos, vocab_size, gap_min, gap_max, training_set = load_project_data()
total_vocab_size, n_embd, n_hidden = vocab_size + 50, 24, 512
BLOCK_SIZE_GEN, BLOCK_SIZE_PRED = 24, 32 # Prevents Shape Errors

# def load_models():
#     gen = Sequential([
#         Embedding(total_vocab_size, n_embd),
#         FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#         FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#         FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#         Linear(n_hidden, vocab_size),
#     ])
#     pred = Sequential([
#         Embedding(total_vocab_size, n_embd),
#         FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#         FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#         FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#         FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#         FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
#         Linear(n_hidden, 1),
#     ])
#     try:
#         g_params = torch.load('notebooks/generator_weights.pt', map_location='cpu')
#         for p, ps in zip(gen.parameters(), g_params): p.data = ps.data
#         p_params = torch.load('notebooks/predictor_weights.pt', map_location='cpu')
#         for p, ps in zip(pred.parameters(), p_params): p.data = ps.data
#         st.sidebar.success("Weights Loaded")
#     except: st.sidebar.error("Weights Not Found")
#     return gen, pred

# gen_model, pred_model = load_models()

st.set_page_config(page_title="QM9-GPT Discovery Lab", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 12px; background-color: #4F46E5; color: white; height: 3.5em; font-weight: bold; }
    .stCode { background-color: #111827; border-radius: 10px; }
    h3 { margin-bottom: 20px; color: #E5E7EB; border-bottom: 1px solid #374151; padding-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_project_data():
    df = pd.read_csv('data/processed/qm9_clean.csv')
    smiles_list = df['smiles'].astype('str').tolist()
    training_set = set(smiles_list)
    chars = sorted(list(set(''.join(smiles_list))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos, len(itos), df['gap_ev'].min(), df['gap_ev'].max(), training_set

stoi, itos, vocab_size, gap_min, gap_max, training_set = load_project_data()
total_vocab_size, n_embd, n_hidden = vocab_size + 50, 24, 512
BLOCK_SIZE_GEN, BLOCK_SIZE_PRED = 24, 32 

def load_models():
    gen = Sequential([
        Embedding(total_vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    pred = Sequential([
        Embedding(total_vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, 1),
    ])
    try:
        g_params = torch.load('outputs/generator_weights.pt', map_location='cpu')
        for p, ps in zip(gen.parameters(), g_params): p.data = ps.data
        p_params = torch.load('outputs/predictor_weights.pt', map_location='cpu')
        for p, ps in zip(pred.parameters(), p_params): p.data = ps.data
        st.sidebar.success("Weights Loaded")
    except: st.sidebar.error("Weights Not Found")
    return gen, pred

gen_model, pred_model = load_models()

st.title("QM9-GPT: Generative Discovery Lab")
st.caption("Targeting specific molecular properties using Transformer-inspired WaveNet architectures.")
st.markdown("---")

with st.sidebar:
    with st.sidebar:
        st.header("Control Panel")
        target_ev = st.slider("Target Energy Gap (eV)", float(gap_min), float(gap_max), 7.0,
                            help="Sets the HOMO-LUMO gap target for the generative model.")
        
        temp = st.select_slider("Sampling Temperature", options=[0.6, 0.7, 0.8, 0.9, 1.0], value=0.8,
                            help="Lower values make the model more predictable; higher values increase novelty but may break chemical rules.")
        generate_btn = st.button("EXECUTE INVERSE DESIGN")

if generate_btn:
    found_valid, attempts = False, 0
    valid_atoms = ['C', 'N', 'O', 'F']

    with st.spinner('Exploring Chemical Space...'):
        while not found_valid and attempts < 25:
            attempts += 1
            b = int(((target_ev - gap_min) / (gap_max - gap_min) * 49))
            context = [vocab_size + b] + [0] * (BLOCK_SIZE_GEN - 1)
            out = []
            while True:
                logits = gen_model(torch.tensor([context]))
                logits = logits if logits.ndim == 2 else logits[:, -1, :]
                if not out:
                    for char, idx in stoi.items():
                        if char not in valid_atoms: logits[0, idx] = -float('inf')
                probs = F.softmax(logits / temp, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                if ix == 0 or len(out) > 60: break
                out.append(itos[ix])
            smiles = "".join(out).replace('.', '')
            mol = Chem.MolFromSmiles(smiles)
            if mol: found_valid = True

    if found_valid:
        is_novel = smiles not in training_set
        encoded = [stoi.get(ch, 0) for ch in smiles]
        padded_pred = encoded[:BLOCK_SIZE_PRED] + [0] * (BLOCK_SIZE_PRED - len(encoded[:BLOCK_SIZE_PRED]))
        pred_val = pred_model(torch.tensor([padded_pred])).item()

        # Row 1: High-Level Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Target Input", f"{target_ev} eV")
        m2.metric("AI Predicted Value", f"{pred_val:.3f} eV", f"{pred_val-target_ev:.2f} diff")
        m3.metric("Discovery Status", "NOVEL" if is_novel else " KNOWN")

        st.markdown("---")
        
        # Row 2: Visualizations
        col_left, col_right = st.columns([1, 1.5])
        
        with col_left:
            st.subheader("2D Topology")
            st.image(Draw.MolToImage(mol, size=(450, 450), wedgeBonds=True), use_container_width=True)
            
            st.markdown("### Chemical Identity")
            st.info(f"**SMILES String:**")
            st.code(smiles, language="text")
            
        with col_right:
            st.subheader("3D Molecular View")
            try:
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                mblock = Chem.MolToMolBlock(mol_3d)
                view = py3Dmol.view(width=700, height=500)
                view.addModel(mblock, 'mol')
                view.setStyle({'stick': {'color':'spectrum', 'radius':0.2}, 'sphere': {'scale': 0.3}})
                view.zoomTo()
                showmol(view, height=500, width=700)
            except:
                st.warning("High-strain detected. Interactive 3D rendering unavailable for this geometry.")
    else:
        st.error("Model failure: No valid chemical syntax found in this sampling batch.")