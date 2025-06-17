import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
from pyswarm import pso  # PSO library

# --- KEY DERIVATION & ENCRYPTION ---
def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_key_file(indices, noise, password: str) -> bytes:
    salt = os.urandom(16)
    key = derive_key(password, salt)
    fernet = Fernet(key)

    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, indices=indices, noise=noise)
    encrypted_data = fernet.encrypt(npz_buf.getvalue())

    return salt + encrypted_data

def decrypt_key_file(data: bytes, password: str):
    salt = data[:16]
    encrypted = data[16:]
    key = derive_key(password, salt)
    fernet = Fernet(key)
    decrypted_npz = fernet.decrypt(encrypted)
    buf = io.BytesIO(decrypted_npz)
    return np.load(buf)

# --- METRICS ---
def calculate_entropy(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 255))
    prob = histogram / np.sum(histogram)
    entropy = -np.sum([p * np.log2(p) for p in prob if p > 0])
    return entropy

def calculate_npcr_uaci(img1, img2):
    diff = img1 != img2
    npcr = np.sum(diff) / diff.size * 100
    uaci = np.mean(np.abs(img1.astype(int) - img2.astype(int)) / 255) * 100
    return npcr, uaci

def calculate_psnr(original, compared):
    mse = np.mean((original - compared) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# --- ENCRYPTION STAGES ---
def logistic_map(x, r=3.99):
    return r * x * (1 - x)

def fractal_encrypt(image, x):
    h, w, c = image.shape
    chaotic_sequence = [logistic_map(x := logistic_map(x)) for _ in range(h * w)]
    chaotic_array = np.array(chaotic_sequence).reshape(h, w)
    chaotic_indices = np.argsort(chaotic_array, axis=1)
    encrypted = np.zeros_like(image)
    for i in range(h):
        encrypted[i] = image[i][chaotic_indices[i]]
    return encrypted, chaotic_indices

def fractal_decrypt(image, chaotic_indices):
    h, w, c = image.shape
    decrypted = np.zeros_like(image)
    for i in range(h):
        decrypted[i][chaotic_indices[i]] = image[i]
    return decrypted

def swarm_encrypt(image, key):
    # PSO to adaptively rearrange pixels
    np.random.seed(key)
    noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)

    # Convert to int16 to avoid overflow, do addition, then back to uint8
    encrypted = (image.astype(np.int16) + noise.astype(np.int16)) % 256
    return encrypted.astype(np.uint8), noise


def swarm_decrypt(image, noise):
    decrypted = (image.astype(np.int16) - noise.astype(np.int16) + 256) % 256
    return decrypted.astype(np.uint8)


def quantum_diffuse(image):
    h, w, c = image.shape
    diffused = image.copy()
    for i in range(h):
        for j in range(w):
            ni, nj = (i + 1) % h, (j + 1) % w
            diffused[i, j] = diffused[i, j] ^ diffused[ni, j] ^ diffused[i, nj]
    return diffused

def reverse_quantum_diffuse(image):
    h, w, c = image.shape
    recovered = image.copy()
    for i in reversed(range(h)):
        for j in reversed(range(w)):
            ni, nj = (i + 1) % h, (j + 1) % w
            recovered[i, j] = recovered[i, j] ^ recovered[ni, j] ^ recovered[i, nj]
    return recovered

# --- PSO-based Swarm Encryption ---
def pso_swarm_encrypt(image, num_particles=10, max_iter=100):
    def fitness_function(permutation):
        permuted_image = image[permutation]
        entropy_val = calculate_entropy(permuted_image)
        return -entropy_val  # Higher entropy is better

    bounds = [(0, image.shape[0]-1)] * image.shape[0]
    best_position, _ = pso(fitness_function, bounds[0], bounds[1], swarmsize=num_particles, maxiter=max_iter)
    return best_position

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("🔐 FSQDE: Fractal-Swarm-Based Quantum Diffusion Encryption")

tab1, tab2 = st.tabs(["🔏 Encrypt Image", "🔓 Decrypt Image"])

with tab1:
    uploaded_file = st.file_uploader("Upload image for encryption", type=["png", "jpg", "jpeg"])
    password = st.text_input("🔑 Set a strong password for encryption", type="password")

    if uploaded_file and password:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Fractal Confusion
        seed = int.from_bytes(password.encode(), 'big') % (10**6)
        x = (seed % 1000) / 1000
        fractal_img, indices = fractal_encrypt(image_np, x)

        # Swarm-Based Permutation using PSO
        swarm_img, noise = swarm_encrypt(fractal_img, seed)

        # Quantum-Inspired Diffusion
        encrypted_img = quantum_diffuse(swarm_img).astype(np.uint8)

        st.image(encrypted_img, caption="🔐 Encrypted Image", use_container_width=True)

        # Metrics
        entropy = calculate_entropy(encrypted_img)
        npcr, uaci = calculate_npcr_uaci(image_np, encrypted_img)
        psnr = calculate_psnr(image_np, encrypted_img)

        st.markdown(f"**Entropy:** {entropy:.4f}")
        st.markdown(f"**NPCR:** {npcr:.2f}%")
        st.markdown(f"**UACI:** {uaci:.2f}%")
        st.markdown(f"**PSNR:** {psnr:.2f} dB")

        # Download encrypted image
        encrypted_pil = Image.fromarray(encrypted_img)
        img_buf = io.BytesIO()
        encrypted_pil.save(img_buf, format="PNG")
        st.download_button("📥 Download Encrypted Image", data=img_buf.getvalue(), file_name="encrypted_fsqde.png")

        # Download encrypted key file
        key_data = encrypt_key_file(indices, noise, password)
        st.download_button("📎 Download Encrypted Key File", data=key_data, file_name="fsqde_key.enc")

with tab2:
    encrypted_file = st.file_uploader("Upload Encrypted Image", type=["png", "jpg", "jpeg"])
    key_file = st.file_uploader("Upload Encrypted Key File (.enc)", type=["enc"])
    password_dec = st.text_input("🔐 Enter password for decryption", type="password")

    if encrypted_file and key_file and password_dec:
        encrypted_image = Image.open(encrypted_file).convert("RGB")
        encrypted_np = np.array(encrypted_image).astype(np.uint8)

        try:
            key_file_data = key_file.read()
            key_data = decrypt_key_file(key_file_data, password_dec)
            indices = key_data["indices"]
            noise = key_data["noise"]

            if encrypted_np.shape != noise.shape:
                st.error("❌ The encrypted image and key file do not match.")
            else:
                reverse_q = reverse_quantum_diffuse(encrypted_np).astype(np.uint8)
                reverse_swarm = swarm_decrypt(reverse_q, noise).astype(np.uint8)
                decrypted_img = fractal_decrypt(reverse_swarm, indices).astype(np.uint8)
                st.image(decrypted_img, caption="✅ Decrypted Image", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error decrypting: {str(e)}")