from parker_cce2022.distill.run_full_space import run_full_space_optimization
from parker_cce2022.distill.run_implicit_function import run_implicit_function_optimization

def main():
    full_space_data = run_full_space_optimization()
    reduced_space_data = run_implicit_function_optimization()
    
    errors = []
    for key in reduced_space_data:
        try:
            abs_err = abs(full_space_data[key] - reduced_space_data[key])
            rel_err = abs_err/max((abs(full_space_data[key]), abs(reduced_space_data[key])))
            errors.append(rel_err)
        except TypeError:
            pass
    
    ave_err = sum(errors)/len(errors)
    
    print("Average relative error:", ave_err)


if __name__ == "__main__":
    main()
