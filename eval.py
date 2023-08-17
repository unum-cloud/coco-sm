import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Union, Optional, List, Callable, Tuple, Dict
from json import load
from os import environ
from importlib import import_module
from enum import Enum
from math import log
from argparse import ArgumentParser

from PIL import Image
from tqdm import tqdm
import pandas as pd


torch.set_grad_enabled(False)
environ['TOKENIZERS_PARALLELISM'] = '(true | false)'


class ReturnType(Enum):
    image = 0
    text = 1
    image_text = 2


class RawDataset(Dataset):
    def __init__(
        self,
        image_dir_path: str,
        meta_path: str,
        return_type: ReturnType,
        lang: str = 'en'
        ):

        super().__init__()
        self.image_dir_path = image_dir_path
        self.return_type = return_type

        with open(meta_path, 'r') as f:
            self.meta = load(f)

        self.lang = lang

    def __getitem__(self, index) -> Union[Image.Image, List[str], Tuple[Image.Image, List[str]]]:
        sample = self.meta[self.lang][index]

        if self.return_type == ReturnType.image:
            return Image.open(f"{self.image_dir_path}/{sample['filename']}")
        elif self.return_type == ReturnType.text:
            return sample['captions']
        
        return Image.open(f"{self.image_dir_path}/{sample['filename']}"), sample['captions']

    def __len__(self):
        return len(self.meta[self.lang])

class Collator:
    def __init__(
        self,
        return_type: ReturnType
        ):

        self.return_type = return_type

    def __call__(self, samples):
        if self.return_type == ReturnType.image:
            return samples
        
        if self.return_type == ReturnType.text:
            return [text for texts in samples for text in texts]
        
        images, text_lists = zip(*samples)

        return images, [text for texts in text_lists for text in texts]


class Evaluate:
    ALL_LANGS = ('en', 'ar', 'de', 'fr', 'he', 'hi', 'hy', 'id', 'it', 'ja', 'ko', 'fa', 'pl', 'pt', 'ru', 'es', 'th', 'tr', 'uk', 'vi', 'zh')
    ID_MAPPING_FILE_PATH = 'meta/id_mapping.json'
    NDCG_AT = 20

    def __init__(
        self,
        image_dir_path: str,
        meta_paths: List[str],
        model: nn.Module,
        image_forward_fn: Callable[[Union[nn.Module, Callable], List[Image.Image], str, Optional[Callable]], torch.Tensor],
        text_forward_fn: Callable[[Union[nn.Module, Callable], List[str], str, Optional[Callable]], torch.Tensor],
        batch_size: int,
        device: str,
        embedding_dim: int,
        image_preprocess: Optional[Callable] = None,
        text_preprocess: Optional[Callable] = None,
        langs: Optional[Tuple[str]] = None
        ):

        self.image_dir_path = image_dir_path
        self.meta_paths = meta_paths
        self.langs = self.ALL_LANGS if langs is None else langs
        self.batch_size = batch_size
        self.device = device
        self.embedding_dim = embedding_dim
        
        with open(self.ID_MAPPING_FILE_PATH) as f:
            data = load(f)
            self.image2text_map = {int(k): v for k, v in data['image2text'].items()}
            self.text2image_map = {int(k): v for k, v in data['text2image'].items()}

        self.model = model.to(device).eval()

        self.image_forward_fn = image_forward_fn
        self.text_forward_fn = text_forward_fn

        self.image_preprocess = image_preprocess
        self.text_preprocess = text_preprocess

        image_dataset = RawDataset(image_dir_path, meta_paths[0], ReturnType.image)

        loader = DataLoader(
            image_dataset,
            batch_size,
            False,
            num_workers=5,
            persistent_workers=True,
            collate_fn=Collator(ReturnType.image)
        )

        self.image_embeddings = torch.empty(len(image_dataset), self.embedding_dim, dtype=torch.float32)
        start_i = 0

        for images in tqdm(loader, desc='Indexing images'):
            embeddings = image_forward_fn(self.model, images, self.device, self.image_preprocess)
            self.image_embeddings[start_i:start_i+embeddings.shape[0]] = embeddings.cpu()
            start_i += embeddings.shape[0]

    def __call__(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        report = []

        for meta_path in self.meta_paths:
            report += self.evaluate(meta_path)

        report = pd.DataFrame(report, index=self.langs * len(self.meta_paths))
        report.index.name = 'language'
        report = report.sort_values('language')
        
        if len(self.meta_paths) == 1:
            return report, None
        group = report.groupby('language')[report.columns[:-1]]
        mean = group.agg('mean').reset_index()
        median_std = mean.iloc[:, 1:].mean().apply(lambda x: f'{x:.3f}') + ' ± ' + mean.iloc[:, 1:].std().apply(lambda x: f'{x:.3f}')
    
        reduced_report = mean.copy(deep=True)
        reduced_report.iloc[:, 1:] = reduced_report.iloc[:, 1:].apply(lambda x: x.apply(lambda n: f'{n:.3f}'))
        reduced_report.loc[reduced_report.shape[0]] = ['all'] + median_std.tolist()

        for meta_path in self.meta_paths:
            meta_src = meta_path.split('/')[-1].replace('.json', '')
            mean = report[report.meta == meta_src].iloc[:, :-1].mean().apply(lambda x: f'{x:.3f}')
            std = report[report.meta == meta_src].iloc[:, :-1].std().apply(lambda x: f'{x:.3f}')
            mean_std = mean + ' ± ' + std

            reduced_report.loc[reduced_report.shape[0]] = [meta_src] + mean_std.tolist()

        return report, reduced_report

    def evaluate(self, meta_path) -> List[Dict[str, Union[str, float]]]:
        meta_src = meta_path.split('/')[-1].replace('.json', '')
        report = []

        t2i_en_ranking = None
        i2t_en_ranking = None

        for lang in self.langs:
            dataset = RawDataset(self.image_dir_path, meta_path, ReturnType.text, lang)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=5,
                persistent_workers=True,
                collate_fn=Collator(ReturnType.text)
            )
            text_embeddings = []

            for texts in tqdm(loader, desc=lang):
                embeddings = self.text_forward_fn(self.model, texts, self.device, self.text_preprocess).cpu()
                text_embeddings.append(embeddings)
            
            text_embeddings = torch.cat(text_embeddings, dim=0)
            text2image_similarity = F.normalize(text_embeddings, dim=1) @ F.normalize(self.image_embeddings, dim=1).t()
            t2i_recall, t2i_ranking = self.calc_recall(text2image_similarity, 'text2image')
            i2t_recall, i2t_ranking = self.calc_recall(text2image_similarity.t(), 'image2text')

            metrics = {f't2i.{k}': v for k, v in t2i_recall.items()}
            metrics.update({f'i2t.{k}': v for k, v in i2t_recall.items()})

            if lang == 'en':
                t2i_en_ranking = t2i_ranking
                i2t_en_ranking = i2t_ranking

            metrics[f't2i.ndcg@{self.NDCG_AT}'] = self.ndcg(t2i_ranking, t2i_en_ranking)
            metrics[f'i2t.ndcg@{self.NDCG_AT}'] = self.ndcg(i2t_ranking, i2t_en_ranking)
            metrics['meta'] = meta_src
            report.append(metrics)

        return report
        
    def calc_recall(self, similarity_matrix: torch.Tensor, mode: str, max_k: int = 10) -> Tuple[Dict[str, float], torch.Tensor]:
        nn_ids = similarity_matrix.topk(max_k, dim=1, largest=True, sorted=True)[1]

        ranks = torch.zeros(similarity_matrix.shape[0])
        labels = self.image2text_map if mode == 'image2text' else self.text2image_map
        for q_id, top in enumerate(nn_ids):
            rank = max_k

            for label in labels[q_id]:
                tmp = torch.where(top == label)[0]

                if len(tmp) != 0:
                    tmp = tmp.item()
                    if tmp < rank:
                        rank = tmp     

            ranks[q_id] = rank
        
        ks = (1, 5, 10)
        recall = {}

        for k in ks:
            recall[f"recall@{k}"] = (ranks < k).float().mean().item()

        return recall, similarity_matrix

    def ndcg(self, similarities, en_similarities):
        similarities = torch.softmax(similarities * 100, dim=1)
        en_similarities = torch.softmax(en_similarities * 100, dim=1)

        def _dcg(ranking, relevance):
            relevance = relevance.gather(1, ranking)
            discounted = relevance / torch.log(torch.arange(2, self.NDCG_AT + 2)) * log(2)
            return discounted.sum(dim=1)

        ranking = similarities.topk(k=self.NDCG_AT, largest=True, sorted=True)[1]
        dcg = _dcg(ranking, en_similarities)

        ideal_ranking = en_similarities.topk(k=self.NDCG_AT, largest=True, sorted=True)[1]
        ideal_dcg = _dcg(ideal_ranking, en_similarities)

        return (dcg / ideal_dcg).mean().item()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--model_name',
        action='store',
        type=str,
        help='model name'
    )

    parser.add_argument(
        '--image_dir_path',
        action='store',
        type=str,
        help='path to the directory with test images'
    )

    parser.add_argument(
        '--meta_files_paths',
        nargs='+',
        help='paths to the files with translations'
    )

    parser.add_argument(
        '--batch_size',
        action='store',
        type=int,
        help='the size of batch'
    )

    parser.add_argument(
        '--device',
        action='store',
        type=str,
        help='the id of device on which the evaluation will be done'
    )

    parser.add_argument(
        '--report_name',
        action='store',
        type=str,
        help='the name of report'
    )

    opts = parser.parse_args()

    test_module = import_module(f'modules.{opts.model_name}')

    evaluate = Evaluate(
        image_dir_path=opts.image_dir_path,
        meta_paths=opts.meta_files_paths,
        model=test_module.model,
        image_forward_fn=test_module.image_forward_fn,
        text_forward_fn=test_module.text_forward_fn,
        batch_size=opts.batch_size,
        device=opts.device,
        embedding_dim=test_module.embedding_dim,
        image_preprocess=test_module.image_preprocess,
        text_preprocess=test_module.text_preprocess
    )

    report, reduced_report = evaluate()

    report.to_csv(f'reports/{opts.report_name}.csv', float_format='%.3f')
    if reduced_report is not None:
        reduced_report.to_csv(
            f'reports/{opts.report_name}_reduced.csv',
            index=False,
            float_format='%.3f'
        )
