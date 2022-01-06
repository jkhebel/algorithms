using Random
using Plots
using ImageView
using TestImages
using ImageCore
using ProgressBars

function flatten(X)
    return reshape(X, (:, last(size(X))))
end

function euclidean_distance(A, B)
    return sum((A-B).^2)^0.5
end


function find_neighbours(p, neighbourhood, ϵ, distanceFunc)
    points = flatten(neighbourhood)
    neighbours::Vector{Int} = []
    for i in range(1, size(points)[1])
        q = points[i, :]
        distance = distanceFunc(p, q)
        if (0 < distance) & (distance <= ϵ)
            push!(neighbours, i)
        end
    end
    return neighbours
end


function dbscan(X, ϵ, minNeighbourhoodSize, distanceFunc)
    points = flatten(X)
    n_points = length(points[:,1])
    cluster::Int = 0
    neighbourhoods = Dict()
    labels = Dict()
    for i in tqdm(range(1, n_points))
        if ~haskey(labels, i)
            get!(neighbourhoods, i) do 
                find_neighbours(points[i,:], points, ϵ, distanceFunc)
            end
            if length(neighbours_i) < minNeighbourhoodSize
                labels[i] = -1 # Noise
            else
                cluster+=1
                labels[i] = cluster

                local_neighbourhood = copy(neighbours_i)
                while ~isempty(local_neighbourhood)
                    j = pop!(local_neighbourhood)
                    if ~haskey(labels, j) || (labels[j] == -1)
                        labels[j] = cluster
                        neighbours_j = find_neighbours(points[j,:], points, ϵ, distanceFunc) # TODO store as a dict
                        if length(neighbours_j) >= minNeighbourhoodSize
                            append!(local_neighbourhood, neighbours_j)
                        end

                    end
                end
            end
        end
    end
    label_map = Array{Int}(undef, n_points, 1)
    for (k, v) in labels
        label_map[k] = v
    end
    label_map = reshape(label_map, size(X)[1:end-1])
    return label_map
end

img = testimage("lighthouse")

data = permutedims(channelview(img), [2,3,1]);
data = convert(Array{Float64, size(data)[end]}, data)

mask = dbscan(data, 0.1, 1000, euclidean_distance)

imshow(mask)