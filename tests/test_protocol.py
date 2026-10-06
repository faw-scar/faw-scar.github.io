import unittest
import numpy as np,pandas as pd,torch
from build_folds import build
from predict import ordinal_probs
from cv_eval import metrics

class ProtocolTests(unittest.TestCase):
    def metadata(self):
        return pd.DataFrame([dict(image_path=f'{plant}_{time}_{view}.jpg',injury_score=1 if time=='7day' else 9,source_set='greenhouse2025',planting='p1',plant_id=str(plant),timepoint=time) for plant in range(4) for time in ['7day','14day'] for view in range(2)])
    def test_timepoints_never_split_physical_plant(self):
        f=build(self.metadata(),n_splits=2)
        self.assertEqual(f.group.nunique(),4)
        for col in ['fold_rep0','fold_rep1']:self.assertTrue(f.groupby('group')[col].nunique().eq(1).all())
    def test_reject_missing_greenhouse_identity(self):
        d=self.metadata();d.loc[0,'plant_id']=''
        with self.assertRaises(ValueError):build(d,2)
    def test_actual_grade_distances_when_middle_grades_absent(self):
        m=metrics(np.array([1,1,9,9]),np.array([1,2,8,9]))
        expected=1-4/sum((a-b)**2 for a in [1,9] for b in [1,2,8,9])
        self.assertAlmostEqual(m['qwk'],expected)
    def test_ordinal_distribution_uses_rank_head(self):
        logits=torch.tensor([[12.,12.,12.,-12.,-12.,-12.,-12.,-12.]])
        probs=ordinal_probs(logits,'frankhall')
        self.assertEqual(int(probs.argmax(1).item())+1,4)
        self.assertTrue(torch.allclose(probs.sum(1),torch.ones(1)))
        self.assertTrue(torch.all(probs>=0))
    def test_nonmonotonic_thresholds_still_normalize(self):
        p=ordinal_probs(torch.tensor([[-2.,2.,-3.,3.,-4.,4.,-5.,5.]]),'frankhall')
        self.assertTrue(torch.all(p>=0));self.assertTrue(torch.allclose(p.sum(1),torch.ones(1)))
if __name__=='__main__':unittest.main()
