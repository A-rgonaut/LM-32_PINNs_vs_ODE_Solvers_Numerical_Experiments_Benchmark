function s = fill_defaults(s, d)
    f = fieldnames(d);
    
    for k = 1:numel(f)
        if ~isfield(s, f{k}) || isempty(s.(f{k}))
            s.(f{k}) = d.(f{k});
        end
    end
end