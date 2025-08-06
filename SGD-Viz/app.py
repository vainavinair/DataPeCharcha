import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from sklearn.datasets import make_regression
from matplotlib import cm
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="SGD vs GD Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class SGDVisualizer:
    def __init__(self, n_samples=1000, noise=0.1, random_state=42):
        np.random.seed(random_state)
        self.n_samples = n_samples
        self.noise = noise
        
        # Generate synthetic dataset for linear regression
        self.X, self.y = make_regression(
            n_samples=n_samples, 
            n_features=2, 
            noise=noise*10, 
            random_state=random_state
        )
        
        # Scale features for better visualization
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        self.y = self.y / np.std(self.y)
        
        # True parameters (analytical solution)
        self.true_params = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        
        # This will hold the shuffled order of data indices for each epoch
        self.indices = np.arange(self.n_samples)

        # Initialize parameters and history
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset parameters and all history to initial values."""
        self.theta = np.array([2.0, -1.5])  # Starting point
        self.sgd_path = [self.theta.copy()]
        self.gd_path = [self.theta.copy()] # For potential future comparison
        self.losses = []
        self.batch_losses = []
        self.current_step = 0
        self.current_epoch = 0
        self.current_batch_in_epoch = 0
        
        # Shuffle indices on reset to ensure the first epoch is random
        np.random.shuffle(self.indices)
    
    def shuffle_data(self):
        """Shuffles the dataset indices for a new epoch and resets the batch counter."""
        np.random.shuffle(self.indices)
        self.current_batch_in_epoch = 0

    def increment_epoch(self):
        """Increments the epoch counter."""
        self.current_epoch += 1

    def compute_loss(self, theta, X=None, y=None):
        """Compute Mean Squared Error loss."""
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        predictions = X @ theta
        return np.mean((predictions - y) ** 2) / 2
    
    def compute_gradient(self, theta, X=None, y=None):
        """Compute gradient of MSE loss."""
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        predictions = X @ theta
        return X.T @ (predictions - y) / len(y)
    
    def create_loss_surface(self, theta_range=(-3, 4), resolution=50):
        """Create loss surface for visualization."""
        theta1_vals = np.linspace(theta_range[0], theta_range[1], resolution)
        theta2_vals = np.linspace(theta_range[0], theta_range[1], resolution)
        
        loss_surface = np.zeros((resolution, resolution))
        
        for i, t1 in enumerate(theta1_vals):
            for j, t2 in enumerate(theta2_vals):
                theta_test = np.array([t1, t2])
                loss_surface[j, i] = self.compute_loss(theta_test)
        
        return theta1_vals, theta2_vals, loss_surface
    
    def create_batch_loss_surface(self, batch_indices, theta_range=(-3, 4), resolution=50):
        """Create loss surface for a specific batch."""
        theta1_vals = np.linspace(theta_range[0], theta_range[1], resolution)
        theta2_vals = np.linspace(theta_range[0], theta_range[1], resolution)
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        batch_loss_surface = np.zeros((resolution, resolution))
        
        for i, t1 in enumerate(theta1_vals):
            for j, t2 in enumerate(theta2_vals):
                theta_test = np.array([t1, t2])
                batch_loss_surface[j, i] = self.compute_loss(theta_test, X_batch, y_batch)
        
        return theta1_vals, theta2_vals, batch_loss_surface
    
    def sgd_step(self, learning_rate, batch_size):
        """Perform one SGD step using pre-shuffled indices."""
        # Calculate slice for the current batch from the shuffled indices
        start_idx = self.current_batch_in_epoch * batch_size
        end_idx = start_idx + batch_size
        
        # Get the actual data indices for the batch
        batch_indices = self.indices[start_idx:end_idx]

        if len(batch_indices) == 0:
            return None, None # Epoch finished

        # Get batch data
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        # Compute gradient on the batch
        grad_batch = self.compute_gradient(self.theta, X_batch, y_batch)
        
        # Update parameters using batch gradient
        self.theta -= learning_rate * grad_batch
        
        # Store path and losses
        self.sgd_path.append(self.theta.copy())
        self.losses.append(self.compute_loss(self.theta)) # Global loss
        self.batch_losses.append(self.compute_loss(self.theta, X_batch, y_batch)) # Batch loss
        
        self.current_step += 1
        self.current_batch_in_epoch += 1 # Prepare for the next step in the epoch
        
        return batch_indices, grad_batch

def create_3d_surface_plot(visualizer, show_batch=False, batch_indices=None):
    """Create 3D surface plot using Plotly."""
    theta1_vals, theta2_vals, loss_surface = visualizer.create_loss_surface()
    
    fig = go.Figure()
    
    # Global loss surface
    fig.add_trace(go.Surface(
        x=theta1_vals, y=theta2_vals, z=loss_surface,
        name='Global Loss Surface', colorscale='Blues', opacity=0.7, showscale=False
    ))
    
    # Batch loss surface (if requested)
    if show_batch and batch_indices is not None and len(batch_indices) > 0:
        _, _, batch_loss_surface = visualizer.create_batch_loss_surface(batch_indices)
        fig.add_trace(go.Surface(
            x=theta1_vals, y=theta2_vals, z=batch_loss_surface,
            name='Batch Loss Surface', colorscale='Reds', opacity=0.6, showscale=False
        ))
    
    # SGD path
    if len(visualizer.sgd_path) > 1:
        path = np.array(visualizer.sgd_path)
        path_losses = [visualizer.compute_loss(theta) for theta in path]
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path_losses,
            mode='lines+markers', name='SGD Path',
            line=dict(color='yellow', width=4), marker=dict(size=3, color='red')
        ))
    
    # Current position
    current_pos = visualizer.sgd_path[-1]
    current_loss = visualizer.compute_loss(current_pos)
    fig.add_trace(go.Scatter3d(
        x=[current_pos[0]], y=[current_pos[1]], z=[current_loss],
        mode='markers', name='Current Position',
        marker=dict(size=8, color='red', symbol='diamond')
    ))
    
    # Global minimum
    fig.add_trace(go.Scatter3d(
        x=[visualizer.true_params[0]], y=[visualizer.true_params[1]],
        z=[visualizer.compute_loss(visualizer.true_params)],
        mode='markers', name='Global Minimum',
        marker=dict(size=8, color='blue', symbol='cross')
    ))
    
    fig.update_layout(
    title='Combined Loss Landscape Contours',
    xaxis_title='θ₁', yaxis_title='θ₂',
    margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_contour_plot(visualizer, show_batch=False, batch_indices=None):
    """Create 2D contour plot."""
    theta1_vals, theta2_vals, loss_surface = visualizer.create_loss_surface()
    
    fig = go.Figure()

    # Global loss contour
    fig.add_trace(go.Contour(
        x=theta1_vals, y=theta2_vals, z=loss_surface,
        name='Global Loss', colorscale='Blues', showscale=False,
        contours=dict(showlabels=True)
    ))

    # Batch loss contour (if requested)
    if show_batch and batch_indices is not None and len(batch_indices) > 0:
        _, _, batch_loss_surface = visualizer.create_batch_loss_surface(batch_indices)
        fig.add_trace(go.Contour(
            x=theta1_vals, y=theta2_vals, z=batch_loss_surface,
            name='Batch Loss', colorscale='Reds', opacity=0.7, showscale=False,
            contours=dict(showlabels=True)
        ))

    # SGD path
    if len(visualizer.sgd_path) > 1:
        path = np.array(visualizer.sgd_path)
        fig.add_trace(go.Scatter(
            x=path[:, 0], y=path[:, 1], mode='lines+markers', name='SGD Path',
            line=dict(color='yellow', width=3), marker=dict(size=4, color='red')
        ))

    fig.update_layout(
        title='Combined Loss Landscape Contours',
        xaxis_title='θ₁', yaxis_title='θ₂',
        width=600, height=500, margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_loss_convergence_plot(visualizer):
    """Create loss convergence plot."""
    if not visualizer.losses:
        return go.Figure().update_layout(title='Loss Convergence', xaxis_title='Step', yaxis_title='Loss')
    
    steps = range(len(visualizer.losses))
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(steps), y=visualizer.losses, name='Global Loss (at step)',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(steps), y=visualizer.batch_losses, name='Batch Loss (at step)',
        line=dict(color='red', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title='Loss Convergence Over Steps',
        xaxis_title='Step', yaxis_title='Loss',
        height=400
    )
    
    return fig

# Main Streamlit App
def main():
    st.title("🎯 Understanding SGD with Batch-wise Loss Landscapes")
    st.markdown("""
    This interactive tool demonstrates how **Stochastic Gradient Descent (SGD)** navigates a "noisy" loss landscape calculated from a small batch of data. 
    Even though the batch landscape (🔴) differs from the true global landscape (🔵), observe how the optimization path (🟡) still progresses towards the global minimum.
    """)
    
    # Initialize session state
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = SGDVisualizer()
        st.session_state.current_batch_indices = None
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Training Parameters")
        
        learning_rate = st.slider(
            "Learning Rate (η)", 
            min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f",
            help="Controls how big of a step the optimizer takes. Larger values converge faster but can be unstable."
        )
        
        batch_size = st.slider(
            "Batch Size", 
            min_value=8, max_value=128, value=32, step=8,
            help="Number of samples used in each SGD step. Smaller batches introduce more noise."
        )
        
        st.header("🎮 Controls")
        
        col1, col2 = st.columns(2)
        
        if col1.button("🚀 Train Epoch", use_container_width=True, help="Run one full pass over the (shuffled) dataset"):
            visualizer = st.session_state.visualizer
            
            # Shuffle data at the start of the epoch's training process
            visualizer.shuffle_data()    
            
            n_batches = (len(visualizer.X) + batch_size - 1) // batch_size
            
            progress_bar = st.progress(0, text=f"Starting Epoch {visualizer.current_epoch + 1}...")
            for i in range(n_batches):
                batch_indices, _ = visualizer.sgd_step(learning_rate, batch_size)
                if batch_indices is not None:
                    st.session_state.current_batch_indices = batch_indices
                
                # This sleep is just to make the progress bar animation visible
                time.sleep(0.05) 
                progress_bar.progress((i + 1) / n_batches, text=f"Epoch {visualizer.current_epoch + 1} - Batch {i+1}/{n_batches}")

            # After the loop, the epoch is done.
            visualizer.increment_epoch() 
            
            st.success(f"Completed Epoch {visualizer.current_epoch}!")
            time.sleep(1) # Pause to show success message
            st.rerun()
    
        if col2.button("🔄 Reset", use_container_width=True, help="Reset the model to its initial state"):
            st.session_state.visualizer.reset_parameters()
            st.session_state.current_batch_indices = None
            st.rerun()

    # Main content tabs
    tab_3d, tab_contour, tab_convergence = st.tabs([
        "🌍 3D Landscape", 
        "📊 Contour View", 
        "📈 Convergence"
    ])
    
    visualizer = st.session_state.visualizer
    batch_indices = st.session_state.current_batch_indices

    # In the main() function

    with tab_3d:
        st.header("3D View: Global vs. Batch Loss Surface")

        # Display the plot first, using the full width and a larger height
        fig_3d = create_3d_surface_plot(visualizer, show_batch=True, batch_indices=batch_indices)
        st.plotly_chart(fig_3d, use_container_width=True, height=700)

        st.markdown("---") # Add a separator

        # Display the metrics below the plot, organized in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Epoch & Step Info**")
            st.metric("Current Epoch", f"{visualizer.current_epoch}")
        with col2:
            if batch_indices is not None:
                st.markdown("**Current Batch Info**")

                current_pos = visualizer.theta
                global_loss = visualizer.compute_loss(current_pos)
                batch_loss = visualizer.compute_loss(
                    current_pos, visualizer.X[batch_indices], visualizer.y[batch_indices]
                )
                st.metric("Global Loss at Current Position", f"{global_loss:.4f}")
                st.metric("Batch Loss at Current Position", f"{batch_loss:.4f}", delta=f"{batch_loss - global_loss:.4f}")
            else:
                st.info("Train an epoch to see batch details.")

    with tab_contour:
        st.header("2D Contour View")
        fig_contour = create_contour_plot(visualizer, show_batch=True, batch_indices=batch_indices)
        st.plotly_chart(fig_contour, use_container_width=True)
        st.markdown("""
        **Key Insight**: The yellow path follows the gradient of the *red* batch surface, not the *blue* global one. The "drunken sailor's walk" of SGD is due to it following these noisy, constantly changing batch landscapes. On average, these steps point towards the true minimum.
        """)

    # In the main() function

    with tab_convergence:
        st.header("Loss Convergence Plot")

        # Display the plot first
        fig_loss = create_loss_convergence_plot(visualizer)
        st.plotly_chart(fig_loss, use_container_width=True)
        
        st.markdown("---") # Add a separator

        # Display metrics below
        st.subheader("Convergence Metrics")
        if visualizer.losses:
            col1, col2, col3 = st.columns(3)
            current_loss = visualizer.losses[-1]
            optimal_loss = visualizer.compute_loss(visualizer.true_params)
            col1.metric("Current Global Loss", f"{current_loss:.6f}")
            col2.metric("Optimal Loss", f"{optimal_loss:.6f}")
            col3.metric("Loss Gap", f"{current_loss - optimal_loss:.6f}")
        else:
            st.info("Train the model to see metrics.")

if __name__ == "__main__":
    main()