"""Research inference with the ordinal branch used by training; no coverage claim."""
import argparse,json
import numpy as np,torch
from PIL import Image,ImageOps
from torchvision import transforms
from faw_model import Net

def ordinal_probs(logits,kind):
    p=torch.sigmoid(logits)
    if kind=='corn':p=torch.cumprod(p,dim=1)
    q=torch.cat([1-p[:,:1],p[:,:-1]-p[:,1:],p[:,-1:]],dim=1).clamp(min=0)
    return q/q.sum(1,keepdim=True).clamp(min=1e-12)

@torch.no_grad()
def predict(checkpoint,image,device=None):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=True)
    meta=checkpoint['meta']
    if meta['two']:raise ValueError('Two-branch inference is not supported by this CLI')
    net=Net(meta['adapt'],False,meta['img_size'])
    net.load_state_dict(checkpoint['state'],strict=False)
    net.eval().to(device)
    tf=transforms.Compose([transforms.Resize((meta['img_size'],meta['img_size'])),transforms.ToTensor(),transforms.Normalize(meta['mean'],meta['std'])])
    with Image.open(image) as im:x=tf(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0).to(device)
    ce,ordinal,binary=net(x)
    p=torch.softmax(ce,1) if meta['predict']=='ce' else ordinal_probs(ordinal,meta['ord'])
    if meta['predict']=='ord' and meta['ord']=='corn':
        grade=int((torch.cumprod(torch.sigmoid(ordinal),1)>.5).sum(1).item())+1
    else:grade=int(p.argmax(1).item())+1
    raw=p[0].cpu().numpy();susceptible=float(raw[4:].sum())
    cv=meta.get('cv_meta',{})
    output=dict(grade=grade,probs=raw.tolist(),tolerance_band=[max(1,grade-1),min(9,grade+1)],tolerance_band_is_coverage_calibrated=False,probabilities='raw model probabilities; independent calibration is not applied by this CLI',injury_category='low injury (Davis 1–4)' if grade<=4 else 'high injury (Davis 5–9)',ordinal_high_injury_probability=susceptible)
    if cv.get('l_bin',0)>0:output['binary_head_high_injury_probability']=float(torch.softmax(binary,1)[0,1].item())
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--image',required=True);ap.add_argument('--device')
    a=ap.parse_args();print(json.dumps(predict(a.checkpoint,a.image,a.device),indent=2))
