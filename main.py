from models.inference import inference

if __name__ == '__main__':
    dataset_path = './dataset/randomizedsmiles.tsv'
    data_id_path = './dataset/randomizedsmiles.id'
    data_file = './dataset/test_spec.mgf'
    out_dir = './results'
    topk = 20
    mode = 'inference'
    inference(dataset_path=dataset_path,
              data_id_path=data_id_path,
              topk=topk,
              data_file=data_file,
              out_dir=out_dir,
              mode=mode)
