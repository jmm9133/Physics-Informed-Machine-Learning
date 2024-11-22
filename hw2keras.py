import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.datasets import mnist
def modified_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] == 0:
            raise ValueError("Matrix is rank deficient.")
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i+1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
            
    return Q, R

def householder_reflection(a):
    a = a.reshape(-1, 1)
    v = a.copy()
    v[0] += np.sign(a[0, 0]) * np.linalg.norm(a)
    v /= np.linalg.norm(v)
    return v

def bidiagonalize(A):
    """
    Reduce matrix A to bidiagonal form B using Householder transformations.
    Returns orthogonal matrices U and V such that A = U * B * V^T
    """
    m, n = A.shape
    A = A.astype(float)
    U = np.eye(m)
    V = np.eye(n)

    for i in range(min(m, n)):
        # Apply Householder to columns (left)
        x = A[i:, i]
        v = householder_reflection(x)
        H = np.eye(m - i) - 2.0 * np.outer(v, v.T)
        A[i:, :] = H @ A[i:, :]
        U[:, i:] = U[:, i:] @ H

        if i < n - 2:
            # Apply Householder to rows (right)
            x = A[i, i+1:].T
            v = householder_reflection(x)
            H = np.eye(n - i - 1) - 2.0 * np.outer(v, v.T)
            A[:, i+1:] = A[:, i+1:] @ H
            V[:, i+1:] = V[:, i+1:] @ H

    return U, A, V


    """
    Compute the SVD of bidiagonal matrix B using the Golub-Kahan algorithm.
    Returns U_svd, S, V_svd such that B = U_svd * S * V_svd^T.
    """
    m, n = B.shape
    U_svd = np.eye(m)
    V_svd = np.eye(n)

    for iteration in range(max_iterations):
        # Check for convergence by examining the off-diagonal elements
        off_diagonal = np.abs(np.diag(B, k=-1))
        if np.all(off_diagonal < epsilon):
            break

        for i in range(n - 1, 0, -1):
            if abs(B[i, i - 1]) < epsilon:
                B[i, i - 1] = 0
            else:
                break

        # Compute Wilkinson's shift
        d = B[n - 2, n - 2]
        f = B[n - 2, n - 1]
        g = B[n - 1, n - 1]
        delta = (d - g) / 2
        sign = np.sign(delta) if delta != 0 else 1
        mu = g - f**2 / (abs(delta) + np.sqrt(delta**2 + f**2)) * sign

        x = B[0, 0]**2 - mu
        z = B[0, 0] * B[0, 1]

        for k in range(n - 1):
            # Left Givens rotation to zero out z
            r = np.hypot(x, z)
            c = x / r
            s = z / r
            G_left = np.array([[c, s], [-s, c]])

            # Apply G_left^T from the left to B
            B[k:k+2, :] = G_left.T @ B[k:k+2, :]

            # Accumulate the rotations into U_svd
            U_svd[:, k:k+2] = U_svd[:, k:k+2] @ G_left

            if k < n - 1:
                x = B[k, k]
                z = B[k, k + 1]

                # Right Givens rotation to zero out z
                r = np.hypot(x, z)
                c = x / r
                s = z / r
                G_right = np.array([[c, s], [-s, c]])

                # Apply G_right from the right to B
                B[:, k:k+2] = B[:, k:k+2] @ G_right

                # Accumulate the rotations into V_svd
                V_svd[:, k:k+2] = V_svd[:, k:k+2] @ G_right

                if k < n - 2:
                    x = B[k, k + 1]
                    z = B[k, k + 2]

    else:
        raise RuntimeError("Golub-Kahan SVD did not converge after maximum iterations.")

    # The singular values are the absolute values of the diagonal elements of B
    S = np.abs(np.diag(B))

    return U_svd, S, V_svd

def streaming_svd(data_generator, k=None):
   
    U = None
    S = None
    Vt = None
    total_m = 0  # Total number of samples processed
    for batch in data_generator:
        n, m_new = batch.shape  # batch shape (n x m_new)
        if U is None:
            # First batch, compute SVD directly
            U, S, Vt = np.linalg.svd(batch)
            if k is not None:
                U = U[:, :k]
                S = S[:k]
                Vt = Vt[:k, :]
            total_m += m_new
        else:
            # Update SVD with new batch
            C = batch  # shape (n x m_new)
            # Step 1: Compute projections P = U^T C
            P = np.dot(U.T, C)  # shape (k x m_new)
            # Step 2: Compute residual R = C - U P
            R = C - np.dot(U, P)  # shape (n x m_new)
            # Step 3: Compute QR decomposition of R
            Q_r, R_r = np.linalg.qr(R)  # Q_r: (n x k_r), R_r: (k_r x m_new)
            # Step 4: Form K matrix
            S_matrix = np.diag(S)  # shape (k x k)
            upper = np.hstack((S_matrix, P))  # shape (k x (k + m_new))
            lower = np.hstack((np.zeros((R_r.shape[0], S_matrix.shape[1])), R_r))  # shape (k_r x (k + m_new))
            K = np.vstack((upper, lower))  # shape ((k + k_r) x (k + m_new))
            # Step 5: Compute SVD of K
            U_k, S_k, Vt_k = np.linalg.svd(K)
            # Step 6: Update U
            U_combined = np.hstack((U, Q_r))  # shape (n x (k + k_r))
            U_new = np.dot(U_combined, U_k)  # shape (n x r_new)
            # Step 7: Update Vt
            # Expand Vt with zeros to match dimensions
            zeros_Vt = np.zeros((Vt.shape[0], m_new))  # shape (k x m_new)
            Vt_expanded = np.hstack((Vt, zeros_Vt))  # shape (k x (total_m + m_new))
            zeros_identity = np.zeros((R_r.shape[0], total_m))  # shape (k_r x total_m)
            identity_R = np.eye(R_r.shape[0])  # shape (k_r x k_r)
            Vt_new_bottom = np.hstack((zeros_identity, identity_R))  # shape (k_r x (total_m + m_new))
            Vt_combined = np.vstack((Vt_expanded, Vt_new_bottom))  # shape ((k + k_r) x (total_m + m_new))
            Vt_new = np.dot(Vt_k, Vt_combined)  # shape (r_new x (total_m + m_new))
            # Step 8: Truncate if necessary
            if k is not None:
                U_new = U_new[:, :k]
                S_k = S_k[:k]
                Vt_new = Vt_new[:k, :]
            U = U_new
            S = S_k
            Vt = Vt_new
            total_m += m_new
    return U, S, Vt

# Example usage (assuming data_generator yields batches of MNIST data)
def data_generator():
    # For demonstration, let's generate random data batches
    n = 784  # Number of features (e.g., flattened 28x28 images)
    batch_size = 50
    num_batches = 10
    for _ in range(num_batches):
        batch = np.random.randn(n, batch_size)
        yield batch



# Step 2: Create data generator
def data_generator_mnist(X, batch_size=50):
    n_samples = X.shape[0]
    num_batches = n_samples // batch_size
    for i in range(num_batches):
        batch = X[i*batch_size:(i+1)*batch_size].T  # Shape (784, batch_size)
        yield batch



# Perform streaming SVD
batch_size = 50
k = 50  # Number of singular values/vectors to keep
#U, S, Vt = streaming_svd(data_generator(), k=k)
from sklearn.datasets import fetch_openml

# Fetch MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#print(mnist)
X = mnist.data  # Shape (70000, 784)

print(X.shape)
y = mnist.target.astype(int)
X = X / 255.0
#batch = data_generator_mnist(X,50)
U, S, Vt = streaming_svd(data_generator_mnist(X, batch_size), k=k)

# Step 4: Visualize the results (optional)
# Plot singular values
plt.figure(figsize=(8, 5))
plt.plot(S, 'o-', linewidth=2)
plt.title('Singular Values of MNIST Dataset')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.grid(True)
plt.show()

# Display the top singular vectors as images
num_components_to_display = 10
fig, axes = plt.subplots(1, num_components_to_display, figsize=(15, 5))
for i in range(num_components_to_display):
    singular_vector = U[:, i]
    image = singular_vector.reshape(28, 28)
    axes[i].imshow(image, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f'Component {i+1}')
plt.show()

# U contains the left singular vectors
# S contains the singular values
# Vt contains the transposed right singular vectors
