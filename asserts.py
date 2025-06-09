def run_asserts():
    import math
    import numpy as np

    assert math.isclose(sig_x_scalar, 0.5, rel_tol=1e-4)
    assert np.allclose(sig_x_array, [0.11920292, 0.26894142, 0.5, 0.73105858, 0.88079708], rtol=1e-4)

    assert math.isclose(derivative_x_scalar, 0.25, rel_tol=1e-4)
    assert np.allclose(derivative_x_array, [0.10499359, 0.19661193, 0.25, 0.19661193, 0.10499359], rtol=1e-4)

    assert np.allclose(vector_word2vec[0], np.array([0.0123291, 0.20410156], dtype=np.float32))
    assert np.allclose(vector_word2vec[1], np.array([0.11230469, -0.06103516], dtype=np.float32))
    assert vector_word2vec[2] is None

    assert np.allclose(vector_glove[0], np.array([0.23088, 0.28283], dtype=np.float32))
    assert np.allclose(vector_glove[1], np.array([-0.215, 0.87737], dtype=np.float32))
    assert vector_glove[2] is None

    assert np.allclose(vector_fasttext[0], np.array([0.047426, -0.042203], dtype=np.float32))
    assert np.allclose(vector_fasttext[1], np.array([0.026351, -0.046308], dtype=np.float32))
    assert vector_fasttext[2] is None

    assert shape_word2vec == [(300,), (300,), None]
    assert shape_glove == [(100,), (100,), None]
    assert shape_fasttext == [(300,), (300,), None]

    assert np.allclose(similarity_w2v, [0.6510956, 0.5854154, 0.1181317])
    assert np.allclose(gensim_similarity_w2v, [0.6510956, 0.5854154, 0.1181317])
    assert np.allclose(similarity_glove, [0.750769, 0.6324707, 0.03835491])
    assert np.allclose(gensim_similarity_glove, [0.750769, 0.6324707, 0.03835491])
    assert np.allclose(similarity_fasttext, [0.77042454, 0.6493072, 0.35930485])
    assert np.allclose(gensim_similarity_fasttext, [0.77042454, 0.6493072, 0.35930485])

    expected_w2v = [
        [('queen', 0.7118193507194519), ('monarch', 0.6189674139022827), ('princess', 0.5902431011199951)],
        [('germany', 0.5094343423843384), ('european', 0.48650455474853516), ('german', 0.4714890420436859)],
        [('gourmet_pizza', 0.5244455337524414), ('gourmet_sandwiches', 0.5183074474334717), ('currywurst', 0.5040446519851685)]
    ]
    for i, (expected, actual) in enumerate(zip(expected_w2v, results_w2v)):
        for j, ((exp_word, exp_score), (act_word, act_score)) in enumerate(zip(expected, actual)):
            assert exp_word == act_word
            assert abs(exp_score - act_score) < 1e-4

    expected_glove = [
        [('queen', 0.7698540687561035), ('monarch', 0.6843381524085999), ('throne', 0.6755736470222473)],
        [('germany', 0.892362117767334), ('austria', 0.7597678303718567), ('poland', 0.7425415515899658)],
        [('gourmet', 0.5839861035346985), ('pastry', 0.5571602582931519), ('dessert', 0.5564545392990112)]
    ]
    for i, (expected, actual) in enumerate(zip(expected_glove, results_glove)):
        for j, ((exp_word, exp_score), (act_word, act_score)) in enumerate(zip(expected, actual)):
            assert exp_word == act_word
            assert abs(exp_score - act_score) < 1e-4

    expected_fasttext = [
        [('queen', 0.7786749005317688), ('queen-mother', 0.7143871784210205), ('king-', 0.6981282234191895)],
        [('germany', 0.6787723898887634), ('poland', 0.6069189310073853), ('europe', 0.601331889629364)],
        [('currywurst', 0.6154085397720337), ('nonkosher', 0.5901243090629578), ('hotdish', 0.5854014754295349)]
    ]
    for i, (expected, actual) in enumerate(zip(expected_fasttext, results_fasttext)):
        for j, ((exp_word, exp_score), (act_word, act_score)) in enumerate(zip(expected, actual)):
            assert exp_word == act_word
            assert abs(exp_score - act_score) < 1e-4

    assert math.isclose(spearman_coeff_word2vec, 0.4420, rel_tol=1e-4)
    assert math.isclose(spearman_coeff_glove, 0.29755, rel_tol=1e-4)
    assert math.isclose(spearman_coeff_fasttext, 0.4409, rel_tol=1e-4)

    print("✅ All tests passed!")
