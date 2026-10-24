"""Small counterexamples for leakage, ranking denominators, and policy gradients."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F

from seqrl.build import prefix
from seqrl.common import seed_all
from seqrl.metrics import per_query,paired_bootstrap
from seqrl.model import create_model,action_logits,next_logits,lm_loss,set_ce_loss,CachedPolicy
from seqrl.reward import RewardFeatures,utility
from seqrl.rl import returns_and_advantages


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_all(19,2)
        cls.config={"vocab_items":24,"context":8,"rl_horizon":5,"sft_layers":2,"sft_heads":4}

    def test_timestamp_ties_are_entirely_invisible(self):
        tokens=np.array([3,4,5,6,7])
        times=np.array([5,10,10,10,15])
        x,seen,count,last=prefix(tokens,times,10,8)
        self.assertEqual(count,1)
        self.assertEqual(last,5)
        self.assertEqual(x.tolist(),[0,0,0,0,0,0,1,3])
        self.assertEqual(seen.tolist(),[3])
        x,seen,count,last=prefix(tokens,times,15,8)
        self.assertEqual(count,4)
        self.assertEqual(set(seen),{3,4,5,6})

    def test_causal_gradient_replay_matches_sequential_sampling(self):
        model=create_model(self.config,16).eval()
        contexts=torch.tensor([[0,0,0,0,1,3,4,5],[0,0,0,1,6,7,8,9]])
        actions=torch.tensor([[7,8,9,10,11],[4,5,6,7,8]])
        replay=action_logits(model,contexts,actions)
        for step in range(5):
            seq=torch.cat([contexts,actions[:,:step]],dim=1)
            self.assertTrue(torch.allclose(replay[:,step],next_logits(model,seq),atol=2e-6))
        changed=actions.clone();changed[:,2:]=20
        self.assertTrue(torch.allclose(replay[:,:3],action_logits(model,contexts,changed)[:,:3],atol=2e-6))
        self.assertTrue((replay[...,:3]<-1e8).all())

    def test_no_gradient_from_masked_ambiguous_labels(self):
        model=create_model(self.config,16).eval()
        x=torch.tensor([[1,3,4,5,6]])
        y=torch.tensor([[-100,-100,-100,5,-100]])
        loss,n=lm_loss(model,x,y)
        expected=F.cross_entropy(next_logits(model,x[:,:3]),torch.tensor([5]))
        self.assertEqual(n,1)
        self.assertTrue(torch.allclose(loss,expected,atol=1e-6))
        none,count=lm_loss(model,x,torch.full_like(x,-100))
        self.assertIsNone(none);self.assertEqual(count,0)

    def test_kv_cache_matches_unpadded_and_padded_full_prefixes(self):
        model=create_model(self.config,16).eval()
        ctx=torch.tensor([[0,0,0,0,1,3,4,5],[1,4,5,6,7,8,9,10]])
        actions=torch.tensor([[3,4,5,6,7],[6,7,8,9,10]])
        cached=CachedPolicy(model,ctx)
        for step in range(5):
            previous=None if step==0 else actions[:,step-1]
            scores=cached.logits(previous)
            full=next_logits(model,torch.cat([ctx,actions[:,:step]],dim=1))
            self.assertTrue(torch.allclose(scores,full,atol=3e-6,rtol=1e-5))

    def test_leave_one_out_baseline_has_correct_discount_and_sign(self):
        r=torch.tensor([[[1.,2.,3.],[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
        returns,adv=returns_and_advantages(r,.5,"append")
        self.assertAlmostEqual(returns[0,0,0].item(),2.75)
        self.assertAlmostEqual(adv[0,0,0].item(),2.25)
        self.assertTrue(torch.allclose(adv.sum(1),torch.zeros((1,3)),atol=1e-6))
        immediate,_=returns_and_advantages(r,.5,"frozen")
        self.assertTrue(torch.equal(immediate,r))
        logits=torch.zeros(4,requires_grad=True)
        loss=-(F.log_softmax(logits,dim=0)*adv[0,:,0].detach()).mean()
        loss.backward()
        self.assertLess(logits.grad[0].item(),0)
        self.assertGreater(logits.grad[1].item(),0)

    def test_oov_targets_reduce_overall_but_not_conditional_quality(self):
        states={"qid":np.arange(2),"uid":np.array([1,2]),"timestamp":np.array([5,5]),
            "all_positives":np.array([2,1]),"positives":np.zeros((2,27),bool),"seen":np.zeros((2,27),bool)}
        states["positives"][0,3]=True
        scores=np.tile(np.r_[-1e9,-1e9,-1e9,np.arange(24,0,-1)],(2,1))
        q,_,_=per_query(states,scores)
        self.assertEqual(q.iloc[0]["recall@10"],.5)
        self.assertAlmostEqual(q.iloc[0]["ndcg@10"],1/(1+1/np.log2(3)))
        self.assertEqual(q.iloc[0]["conditional_ndcg@10"],1.)
        self.assertEqual(q.iloc[0]["map@10"],.5)
        self.assertEqual(q.iloc[1]["ndcg@10"],0.)
        self.assertTrue(np.isnan(q.iloc[1]["conditional_ndcg@10"]))
        self.assertEqual(paired_bootstrap(q,q,"ndcg@10",100)["ci95"],[0.,0.])

    def test_rm_fraction_does_not_count_bos_as_an_item(self):
        f=RewardFeatures.__new__(RewardFeatures)
        f.embedding=np.zeros((27,2),np.float32)
        f.duration=np.ones(27);f.popularity=np.ones(27)
        x=np.array([[0,0,1,3,2,4],[3,4,5,6,7,8]])
        p=f.profiles(x,np.zeros((2,27),bool),np.array([3,6]))
        self.assertAlmostEqual(p["known_fraction"][0],2/3)
        self.assertEqual(p["known_fraction"][1],1)

    def test_reward_objectives_distinguish_duration_and_repetition(self):
        a=np.array([3,4]);ctx=np.array([[1,3],[1,3]])
        pred={"completion":np.ones(2)*.8,"return_proxy":np.ones(2)*.5,"uncertainty":np.zeros(2)}
        cat={"duration":np.r_[0,0,0,100,300],"popularity":np.r_[0,0,0,5,5]}
        seconds=utility(pred,a,ctx,cat,"seconds")
        self.assertAlmostEqual(seconds[1]/seconds[0],3)
        balanced=utility(pred,a,ctx,cat,"balanced")
        self.assertAlmostEqual(balanced[1]-balanced[0],.25)


if __name__=="__main__":
    unittest.main()
