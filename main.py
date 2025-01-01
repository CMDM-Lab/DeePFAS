from models.inference import inference

if __name__ == '__main__':
    dataset_path = './dataset/smiles_dataset.tsv'
    data_id_path = './dataset/smiles_dataset.id'
    data_file = './example/testdata.mgf'
    out_dir = './results'
    topk = 20
    mode = 'inference'
    inference(dataset_path=dataset_path,
              data_id_path=data_id_path,
              topk=topk,
              data_file=data_file,
              out_dir=out_dir,
              mode=mode)
