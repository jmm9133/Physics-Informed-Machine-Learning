import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import time
import matplotlib.animation as animation
from IPython.display import HTML
def streaming_svd(data_generator, k=None,ff=1):
   
    U = None
    S = None
    Vt = None
    total_m = 0  # Total number of samples processed
    singular_vectors_over_time = []
    for batch in data_generator:
        if U is None:
            n, m_new = batch.shape  # batch shape (n x m_new)
            A = batch
            Q,R = np.linalg.qr(A)
            U_batch,S_batch,Vt_batch = np.linalg.svd(R)
            total_m += 1
            if k is not None:
                K = min(k, len(S_batch))
            else:
                K = len(S_batch)
            U = Q @ U_batch[:, :K]
            S = S_batch[:K]
        else:
            Ai = batch
            m_new = Ai.shape[1]
            # Compute ff * U @ diag(S)
            U_S = ff * U @ np.diag(S)

            # Concatenate along columns
            M = np.hstack([U_S, Ai])

            # QR decomposition
            Q, R = np.linalg.qr(M,mode='reduced')

            # SVD of R
            U_tilde, D_tilde, Vt_tilde = np.linalg.svd(R,full_matrices=False)

            # Determine number of components to retain
            if k is not None:
                K = min(k, len(D_tilde))
            else:
                K = len(D_tilde)

            U_tilde_K = U_tilde[:, :K]
            D_tilde_K = D_tilde[:K]

            # Update U and S
            U = Q @ U_tilde_K
            S = D_tilde_K
            total_m += 1
            singular_vectors_over_time.append(U[:, :3].copy())

    return U, S, Vt,singular_vectors_over_time

def data_generator_mnist(X, batch_size=50):
    n_samples = X.shape[0]
    num_batches = n_samples // batch_size
    for i in range(num_batches):
        batch = X[i*batch_size:(i+1)*batch_size].T  # Shape (784, batch_size)
        yield batch
def measure_streaming_svd_time(data_generator_mnist, streaming_svd, X, batch_sizes, k=50):
    avg_times = {}
    
    for batch_size in batch_sizes:
        times = []
        for _ in range(10):  # Run each batch size 10 times
            start_time = time.time()
            
            # Run streaming SVD with the given batch size and rank k
            U, S, Vt = streaming_svd(data_generator_mnist(X, batch_size), k=batch_size)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate and store the average time for the current batch size
        avg_times[batch_size] = np.mean(times)
        print(f"Batch size {batch_size}: Average Time = {avg_times[batch_size]:.4f} seconds")

    return avg_times

def visualize_singular_vectors(singular_vectors_over_time, image_shape=(28, 28)):
    num_batches = len(singular_vectors_over_time)
    num_components = 3  # We are interested in the first three singular vectors

    fig, axes = plt.subplots(1, num_components, figsize=(12, 4))

    ims = []

    for t in range(num_batches):
        images = []
        for i in range(num_components):
            singular_vector = singular_vectors_over_time[t][:, i]
            image = singular_vector.reshape(image_shape)
            im = axes[i].imshow(image, animated=True, cmap='gray')
            axes[i].set_title(f'Component {i+1}')
            axes[i].axis('off')
            images.append(im)
        ims.append(images)

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    ani.save('singular_vectors_evolution.mp4')
    plt.close(fig)
    print("Animation saved as 'singular_vectors_evolution.mp4'")

def compare_singular_vectors(U_streaming, U_full, num_components=3):
    for i in range(num_components):
        vec_streaming = U_streaming[:, i]
        vec_full = U_full[:, i]

        # Compute cosine similarity
        cosine_similarity = np.abs(np.dot(vec_streaming, vec_full)) / (np.linalg.norm(vec_streaming) * np.linalg.norm(vec_full))
        print(f"Component {i+1} Cosine Similarity: {cosine_similarity:.4f}")


    
from sklearn.datasets import fetch_openml





# Fetch MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#print(mnist)
X = mnist.data  # Shape (70000, 784)
batch_size = 50
k = None
print(X.shape)
y = mnist.target.astype(int)
X = X / 255.0
#batch = data_generator_mnist(X,50)
#U,S,Vt = np.linalg.svd(X)
#U, S, Vt = streaming_svd(data_generator_mnist(X, batch_size), k=50)

## Part 2 Batches

batch_sizes = [10,20,30,40,50,60,70,80,90,100]

#average_times = measure_streaming_svd_time(data_generator_mnist,streaming_svd,X,batch_sizes,k=50)
# keys = list(average_times.keys())
# values = list(average_times.values())
# plt.figure(figsize=(8, 5))
# plt.bar(keys, values)
# plt.title('Run time based on batch size')
# plt.xlabel('Batch Size')
# plt.ylabel('Run Time')
# plt.grid(True)
# plt.show()
ff = 1.0
U, S, Vt, singular_vectors_over_time = streaming_svd(data_generator_mnist(X, batch_size), k=50, ff=ff)

# Visualize the evolution
visualize_singular_vectors(singular_vectors_over_time, image_shape=(28, 28))

# Compute full SVD using np.linalg.svd
U_full, S_full, Vt_full = np.linalg.svd(X.T, full_matrices=False)

# Compare singular vectors for different forget-factors
forget_factors = [1.0, 0.9, 0.8, 0.7,.6,.5,.4,.3,.2,.1]

for ff in forget_factors:
    print(f"\nForget-factor: {ff}")
    U_streaming, S_streaming, Vt_streaming, _ = streaming_svd(data_generator_mnist(X, batch_size), k=50, ff=ff)
    compare_singular_vectors(U_streaming, U_full, num_components=3)

# # Display the top singular vectors as images
# num_components_to_display = 10
# fig, axes = plt.subplots(1, num_components_to_display, figsize=(15, 5))
# for i in range(num_components_to_display):
#     singular_vector = U[:, i]
#     image = singular_vector.reshape(28, 28)
#     axes[i].imshow(image, cmap='gray')
#     axes[i].axis('off')
#     axes[i].set_title(f'Component {i+1}')
# plt.show()