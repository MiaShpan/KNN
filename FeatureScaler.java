package HomeWork3;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.Filter;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 *
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public Instances scaleData(Instances instances) {
		final Standardize standardizeFilter = new Standardize();
		try {
			standardizeFilter.setInputFormat(instances);
			return Filter.useFilter(instances, standardizeFilter);
		} catch (Exception e) {
			return null;
		}
	}
}