import sys
from pathlib import Path
 
sys.path.append(str(Path(__file__).resolve().parent.parent))
 
import pandas as pd
 
from features.engineer    import add_features, prepare_Xy, time_split
from features.feature_store import save_features, load_features, features_exist, delete_features
from config import TEST_SIZE


class FeaturePipeline:
    
    def __init__(self, force_recompute: bool = False, test_size = TEST_SIZE):
        
        self.force_recompute = force_recompute
        self.test_size = TEST_SIZE 
        
    def run(self, df_raw: pd.DataFrame) -> DataFrame:
        
        if not self.force_recompute and features_exist():
            print("[FeaturePipeline] Loading features from the store (use force_recmpute= True to refresh).")
        
        print("[FeaturePipeline] Computing features from raw data...")
        
        df_feat = add_features(df_raw)
        save_features(df_feat)
        
        return df_feat
    
    def prepare(self, df_feat: pd.DataFrame):
        
        return prepare_Xy(df_feat)
    
    def split(self, X: pd.DataFrame, y: pd.Series):
        
        return time_split(X, y, test_size= self.test_size)
    
    def full_run(self, df_raw: pd.DataFrame):
        
        df_feat = self.run(df_raw)
        X, y, df_feat = self.prepare(df_feat)
        X_train, X_test, y_train, y_test = self.split(X, y)
        
        return X_train, X_test, y_train, y_test, df_feat
    
    def invalidate_cache(self) -> None:
        
        delete_features()