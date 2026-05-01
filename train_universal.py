import pandas as pd
import logging
from model import CableFaultDetector
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    print("Loading Universal Multi-Modal Dataset...")
    try:
        df_elec = pd.read_csv("datasets/realistic_data.csv")
        df_elec["cable_domain_id"] = 0
    except FileNotFoundError:
        df_elec = pd.DataFrame()
        
    try:
        df_opt = pd.read_csv("datasets/optical_240km.csv")
        df_opt["cable_domain_id"] = 1
    except FileNotFoundError:
        df_opt = pd.DataFrame()
        
    df_combined = pd.concat([df_elec, df_opt], ignore_index=True)
    
    if len(df_combined) == 0:
        print("No datasets found!")
        return
        
    print(f"Combined dataset size: {len(df_combined)} samples")
    print("Training Universal Model...")
    
    detector = CableFaultDetector()
    detector.train(df_combined)
    
    result = detector.predict(df_combined)
    metrics = detector.evaluate(result)
    
    detector.save()
    with open("saved_model/roc_auc.pkl", "wb") as f:
        pickle.dump(metrics["roc_auc"], f)
        
    print("Universal Training Complete!")

if __name__ == "__main__":
    main()
