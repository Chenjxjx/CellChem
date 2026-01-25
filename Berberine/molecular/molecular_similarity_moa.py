import pandas as pd
import os

INPUT_FILE = "berberine_similarity_with_moa.csv"
OUTPUT_FILE = "top_50_moas_by_appearance.csv"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run the similarity search script first.")
        return

    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values('pearson', ascending=False)
    
    print(f"Loaded {len(df)} records.")
    
    ordered_moas = []
    seen_moas = set()
    
    moa_details = []
    
    for rank, row in df.iterrows():
        current_rank = list(df.index).index(rank) + 1
        
        raw_moa = row['moa']
        if pd.isna(raw_moa): continue
        if "Target Molecule" in str(raw_moa): continue
        
        moas = [m.strip() for m in str(raw_moa).split('|')]
        
        for m in moas:
            if m not in seen_moas and m != "nan":
                seen_moas.add(m)
                ordered_moas.append(m)
                
                moa_details.append({
                    "Order": len(ordered_moas),
                    "MOA": m,
                    "First_Appeared_In_Drug": row['pert_id'],
                    "Drug_Rank": current_rank,
                    "Similarity_Score": row['pearson']
                })
                
            if len(ordered_moas) >= 50:
                break
        
        if len(ordered_moas) >= 50:
            break

    result_df = pd.DataFrame(moa_details)
    result_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n--- Top 50 Unique MOAs (By Appearance Order) ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)
    
    print(result_df[['Order', 'MOA', 'Drug_Rank', 'Similarity_Score']])
    print(f"\nFull list saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
