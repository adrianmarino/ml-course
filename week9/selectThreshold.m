function [bestEpsilon bestF1] = selectThreshold(yval, pval)
    bestEpsilon = 0;
    bestF1 = 0;
    F1 = 0;
    stepsize = (max(pval) - min(pval)) / 1000;

    for epsilon = min(pval):stepsize:max(pval)
        tp = sum((yval == 1) & (pval < epsilon));
        fp = sum((yval == 0) & (pval < epsilon));
        fn = sum((yval == 1) & (pval >= epsilon));

        if (tp + fp == 0 || tp + fn == 0)
            continue
        end
        
        prec = tp / (tp + fp);
        rec = tp / (tp + fn);
        F1 = (2 * prec * rec) / (prec + rec);

        if F1 > bestF1
            bestF1 = F1;
            bestEpsilon = epsilon;
        end
    end
end
