"""
AI Agent for User Clustering with Energy Optimization
Protocol: 80% Cosine Similarity Threshold
Author: AI Clustering Protocol v1.0
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from codecarbon import EmissionsTracker
import warnings
warnings.filterwarnings('ignore')

class UserClusteringAgent:
    """
    AI Agent that clusters users based on similarity to reduce computational load
    """
    
    def __init__(self, similarity_threshold=0.8, min_cluster_size=10, max_cluster_size=500):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.scaler = StandardScaler()
        self.clusters = None
        self.baseline_user_count = 0
        self.new_users_count = 0
        self.last_clustering_timestamp = None
        
        # Re-clustering triggers
        self.triggers = {
            'new_users_threshold': 0.10,  # 10%
            'quality_threshold': 0.3,     # Silhouette score
            'time_based_days': 7
        }
        
        print("🤖 AI Clustering Agent initialized")
        print(f"   Similarity threshold: {similarity_threshold}")
        print(f"   Cluster size: {min_cluster_size}-{max_cluster_size} users\n")
    
    def generate_synthetic_users(self, n_users=100):
        """Generate synthetic e-commerce user data"""
        np.random.seed(42)
        
        data = {
            'user_id': [f'user_{i}' for i in range(n_users)],
            'age': np.random.randint(18, 68, n_users),
            'location': np.random.choice(['US', 'UK', 'DE', 'FR', 'CA'], n_users),
            'posts_per_week': np.random.randint(0, 20, n_users),
            'likes_per_week': np.random.randint(0, 100, n_users),
            'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_users),
            'interest_electronics': np.random.randint(0, 2, n_users),
            'interest_fashion': np.random.randint(0, 2, n_users),
            'interest_home': np.random.randint(0, 2, n_users),
            'interest_sports': np.random.randint(0, 2, n_users),
        }
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {n_users} synthetic users")
        return df
    
    def prepare_features(self, df):
        """Convert user data to feature vectors"""
        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df, columns=['location', 'device'], drop_first=True)
        
        # Select feature columns (exclude user_id)
        feature_cols = [col for col in df_encoded.columns if col != 'user_id']
        X = df_encoded[feature_cols].values
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X)
        
        return X_normalized, df_encoded['user_id'].values
    
    def calculate_similarity_matrix(self, X):
        """Calculate cosine similarity between all users"""
        similarity_matrix = cosine_similarity(X)
        return similarity_matrix
    
    def perform_clustering(self, X, n_clusters=5):
        """Perform K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        return labels, kmeans
    
    def assign_clusters_by_similarity(self, X, labels, user_ids):
        """Assign users to clusters only if similarity >= threshold"""
        similarity_matrix = self.calculate_similarity_matrix(X)
        
        results = []
        for i, user_id in enumerate(user_ids):
            cluster_id = labels[i]
            
            # Find all users in the same cluster
            cluster_members = np.where(labels == cluster_id)[0]
            
            # Calculate average similarity to cluster members
            if len(cluster_members) > 1:
                similarities = similarity_matrix[i, cluster_members]
                avg_similarity = np.mean(similarities[similarities != 1.0])  # Exclude self
            else:
                avg_similarity = 0
            
            # Assign to cluster only if similarity >= threshold
            if avg_similarity >= self.similarity_threshold:
                final_cluster = cluster_id
                status = 'clustered'
            else:
                final_cluster = -1  # Outlier
                status = 'outlier'
            
            results.append({
                'user_id': user_id,
                'cluster': final_cluster,
                'similarity': avg_similarity,
                'status': status
            })
        
        return pd.DataFrame(results)
    
    def validate_cluster_sizes(self, results_df):
        """Validate that clusters meet size requirements"""
        cluster_sizes = results_df[results_df['cluster'] != -1]['cluster'].value_counts()
        
        valid_clusters = []
        for cluster_id, size in cluster_sizes.items():
            if self.min_cluster_size <= size <= self.max_cluster_size:
                valid_clusters.append(cluster_id)
        
        # Mark invalid cluster members as outliers
        results_df.loc[~results_df['cluster'].isin(valid_clusters + [-1]), 'cluster'] = -1
        results_df.loc[results_df['cluster'] == -1, 'status'] = 'outlier'
        
        return results_df
    
    def calculate_quality_metrics(self, X, labels):
        """Calculate clustering quality metrics"""
        # Filter out outliers for silhouette score
        valid_indices = labels != -1
        
        if len(np.unique(labels[valid_indices])) < 2:
            return {'silhouette_score': 0}
        
        silhouette = silhouette_score(X[valid_indices], labels[valid_indices])
        
        return {
            'silhouette_score': silhouette
        }
    
    def check_reclustering_triggers(self, quality_metrics):
        """Check if re-clustering should be triggered"""
        triggers_activated = []
        
        # Check new users threshold
        if self.baseline_user_count > 0:
            new_users_percent = (self.new_users_count / self.baseline_user_count)
            if new_users_percent >= self.triggers['new_users_threshold']:
                triggers_activated.append(f"New users: {new_users_percent*100:.1f}%")
        
        # Check quality threshold
        if quality_metrics['silhouette_score'] < self.triggers['quality_threshold']:
            triggers_activated.append(f"Low quality: {quality_metrics['silhouette_score']:.3f}")
        
        return triggers_activated
    
    def run_clustering(self, df, n_clusters=5):
        """Main clustering pipeline with carbon tracking"""
        print("\n🚀 Starting clustering process...")
        
        # Start carbon tracking
        tracker = EmissionsTracker(project_name="user_clustering")
        tracker.start()
        
        try:
            # Prepare features
            X, user_ids = self.prepare_features(df)
            print(f"📊 Prepared {X.shape[1]} features for {len(user_ids)} users")
            
            # Perform clustering
            labels, kmeans_model = self.perform_clustering(X, n_clusters)
            print(f"🔍 K-means clustering complete ({n_clusters} clusters)")
            
            # Assign clusters based on similarity threshold
            results_df = self.assign_clusters_by_similarity(X, labels, user_ids)
            
            # Validate cluster sizes
            results_df = self.validate_cluster_sizes(results_df)
            
            # Calculate metrics
            clustered_users = results_df[results_df['status'] == 'clustered']
            outliers = results_df[results_df['status'] == 'outlier']
            
            print(f"\n✅ Clustering Results:")
            print(f"   👥 Clustered users: {len(clustered_users)}")
            print(f"   ⚠️  Outliers: {len(outliers)}")
            print(f"   📈 Average similarity: {clustered_users['similarity'].mean():.3f}")
            
            # Quality metrics
            quality_metrics = self.calculate_quality_metrics(X, results_df['cluster'].values)
            print(f"   📊 Silhouette score: {quality_metrics['silhouette_score']:.3f}")
            
            # Check re-clustering triggers
            triggers = self.check_reclustering_triggers(quality_metrics)
            if triggers:
                print(f"\n⚠️  Re-clustering triggers activated:")
                for trigger in triggers:
                    print(f"      • {trigger}")
            
            # Update baseline
            self.baseline_user_count = len(df)
            self.new_users_count = 0
            self.clusters = results_df
            
            # Stop carbon tracking
            emissions = tracker.stop()
            
            # Calculate energy savings
            individual_processing = len(df)  # Each user processed individually
            cluster_processing = len(clustered_users['cluster'].unique())
            energy_saved = ((individual_processing - cluster_processing) / individual_processing) * 100
            
            print(f"\n🌱 Carbon Tracking (CodeCarbon):")
            print(f"   CO2 emissions: {emissions:.6f} kg")
            print(f"   Energy saved: {energy_saved:.1f}%")
            print(f"   Processing: {cluster_processing} clusters vs {individual_processing} individuals\n")
            
            return results_df, quality_metrics
            
        except Exception as e:
            tracker.stop()
            print(f"❌ Error during clustering: {e}")
            raise
    
    def add_new_users(self, existing_df, n_new_users=15):
        """Simulate adding new users"""
        new_users = self.generate_synthetic_users(n_new_users)
        
        # Adjust user IDs to avoid conflicts
        max_id = int(existing_df['user_id'].str.replace('user_', '').max())
        new_users['user_id'] = [f'user_{max_id + i + 1}' for i in range(n_new_users)]
        
        combined_df = pd.concat([existing_df, new_users], ignore_index=True)
        self.new_users_count += n_new_users
        
        print(f"\n➕ Added {n_new_users} new users (total: {len(combined_df)})")
        print(f"   New users since last clustering: {self.new_users_count} ({(self.new_users_count/self.baseline_user_count)*100:.1f}%)")
        
        return combined_df
    
    def export_results(self, results_df, filename='cluster_results.csv'):
        """Export clustering results to CSV"""
        results_df.to_csv(filename, index=False)
        print(f"💾 Results exported to {filename}")


def main():
    """Main execution function"""
    print("=" * 60)
    print("AI CLUSTERING AGENT - E-COMMERCE USER OPTIMIZATION")
    print("Protocol: 80% Cosine Similarity | Min Cluster: 10 users")
    print("=" * 60)
    
    # Initialize agent
    agent = UserClusteringAgent(similarity_threshold=0.8, min_cluster_size=10)
    
    # Generate initial users
    users_df = agent.generate_synthetic_users(n_users=100)
    
    # Run initial clustering
    results, metrics = agent.run_clustering(users_df, n_clusters=5)
    
    # Show cluster distribution
    print("\n📊 Cluster Distribution:")
    cluster_dist = results[results['status'] == 'clustered']['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_dist.items():
        print(f"   Cluster {cluster_id}: {count} users")
    
    # Simulate adding new users (triggering re-clustering)
    print("\n" + "="*60)
    print("SIMULATING NEW USER GROWTH")
    print("="*60)
    
    users_df = agent.add_new_users(users_df, n_new_users=15)
    
    # Check if re-clustering is needed
    if agent.new_users_count / agent.baseline_user_count >= agent.triggers['new_users_threshold']:
        print("\n⚠️  New users threshold exceeded! Re-clustering recommended...")
        results, metrics = agent.run_clustering(users_df, n_clusters=5)
    
    # Export results
    agent.export_results(results)
    
    print("\n✅ Process complete!")


if __name__ == "__main__":
    main()
