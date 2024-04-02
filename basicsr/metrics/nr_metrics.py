def calculate_musiq(img, musiq_fn=None):
    
    assert musiq_fn is not None

    img = img.cuda()
    
    musiq_result = musiq_fn(img).item()

    return musiq_result

def calculate_maniqa(img, maniqa_fn=None):
    
    assert maniqa_fn is not None

    img = img.cuda()
    
    maniqa_result = maniqa_fn(img).item()

    return maniqa_result
