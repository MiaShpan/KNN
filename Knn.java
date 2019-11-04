package HomeWork3;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class DistanceCalculator {
    /**
    * We leave it up to you wheter you want the distance method to get all relevant
    * parameters(lp, efficient, etc..) or have it has a class variables.
    */
    public double distance (Instance one, Instance two, int lpDistance) {
        // the lp value is not infinity
        if (lpDistance != 0) {
            return lpDistance(one, two, lpDistance);
        } else {
            // the lp value is infinity
            return lInfinityDistance(one, two);
        }
    }
    public double distance (Instance one, Instance two, int lpDistance, double currentKthDistance) {
        // the lp value is not infinity
        if (lpDistance != 0) {
            return efficientLpDistance(one, two, lpDistance, currentKthDistance);
        } else {
            // the lp value is infinity
            return efficientLInfinityDistance(one, two, currentKthDistance);
        }
    }

    /**
     * Returns the Lp distance between 2 instances.
     * @param one
     * @param two
     */
    private double lpDistance(Instance one, Instance two, int p) {
      double sum = 0;
      double currentAttributeValue1;
      double currentAttributeValue2;

      for(int i = 0; i < one.numAttributes() - 1 ; i++){
          currentAttributeValue1 = one.value(i);
          currentAttributeValue2 = two.value(i);
          // (x1(i) - x2(i))^p
          sum += Math.pow(Math.abs(currentAttributeValue1 - currentAttributeValue2),p);
      }
      return Math.pow(sum, 1/p);
    }

    /**
     * Returns the L infinity distance between 2 instances.
     * @param one
     * @param two
     * @return
     */
    private double lInfinityDistance(Instance one, Instance two) {
        double currentAttributeDistance;
        double max = 0;
        // for every attribute
        for (int i = 0; i < one.numAttributes() - 1; i++){
            // calculates the distance between the two attribute
            currentAttributeDistance = Math.abs(one.value(i) - two.value(i));
            // if the current distance is bigger than the maximum found so far
            if (currentAttributeDistance > max){
                // current distance is the max found so far
                max = currentAttributeDistance;
            }
        }
        return max;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLpDistance(Instance one, Instance two, int p, double currentKthDistance) {
        double sum = 0;
        double currentAttributeValue1;
        double currentAttributeValue2;

        for(int i = 0; i < one.numAttributes() - 1 ; i++){
            currentAttributeValue1 = one.value(i);
            currentAttributeValue2 = two.value(i);
            // (x1(i) - x2(i))^p
            sum += Math.pow(Math.abs(currentAttributeValue1 - currentAttributeValue2),p);
            // efficient - stops if the sum is larger than the current kth distance
            if(sum >= Math.pow(currentKthDistance,p))
            {
                return -1;
            }
        }
        return Math.pow(sum, 1/p);
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     * @param one
     * @param two
     * @return
     */
    private double efficientLInfinityDistance(Instance one, Instance two, double currentKthDistance) {
        double currentAttributeDistance;
        double max = 0;
        // for every attribute
        for (int i = 0; i < one.numAttributes() - 1; i++){
            // calculates the distance between the two attribute
            currentAttributeDistance = Math.abs(one.value(i) - two.value(i));
            // efficient - stops if the distance is larger tham the current kth distance
            if (currentAttributeDistance >= currentKthDistance)
            {
                return -1;
            }
            // if the current distance is bigger than the maximum found so far
            if (currentAttributeDistance > max){
                // current distance is the max found so far
                max = currentAttributeDistance;
            }
        }
        return max;
    }
}

public class Knn implements Classifier {

    public enum DistanceCheck{Regular, Efficient}
    public DistanceCheck m_distanceCheck;
    private Instances m_trainingInstances;
    private double[] m_distances;
    public int m_lpdistance;
    // true if uniform
    public boolean isUniform;
    public int m_k;
    private DistanceCalculator distanceCalculator;
    private int[] m_kNearestNeighbors;

    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        m_trainingInstances = instances;
    }

    // sets the k,p and weight scheme values of this combination
    public void setCombination(int p, int k, boolean weightingScheme){
        m_lpdistance = p;
        m_k = k;
        isUniform = weightingScheme;
        distanceCalculator = new DistanceCalculator();
        m_distanceCheck = DistanceCheck.Regular;
    }

    // sets the k,p, weight scheme and distance check values of this combination
    public void setCombination(int p, int k, boolean weightingScheme, DistanceCheck dc){
        m_lpdistance = p;
        m_k = k;
        isUniform = weightingScheme;
        distanceCalculator = new DistanceCalculator();
        m_distanceCheck = dc;
    }

    /**
     * Returns the knn prediction on the given instance.
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        // if we want to use the original data
        findNearestNeighbors(instance);
        if(isUniform){
            return getAverageValue();
        }
        // if we want to use weight data
        return getWeightedAverageValue();
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     * @return
     */
    public double calcAvgError (Instances validationSet){

        double realValue;
        double prediction;
        double sum = 0;

        // for every instance in the validation set
        for (int i = 0; i < validationSet.numInstances(); i++){
            // get the real value of the class
            realValue = validationSet.instance(i).classValue();
            // get our predication
            prediction = regressionPrediction(validationSet.instance(i));
            // the mistake is the distance between the real value to the prediction
            sum += Math.abs(realValue - prediction);
        }
        // calculate the average of the mistakes
        return sum/validationSet.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     * @param num_of_folds
     * @param instances
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) throws Exception {
        // instances needed to be speared in the folds
        int numOfInstancesToSplit = instances.numInstances();
        // num of instances in every fold (more or less)
        int predictedFoldSize = instances.numInstances()/num_of_folds;
        // we want to start from 0
        int foldLastIndex = -1;
        int currentFoldSize;
        double sumOfErrors = 0;
        Instances currentFold;
        Instances currentFoldTraining;

        for (int fold = 0; fold < num_of_folds; fold++) {
            // creates a new group of instances
            currentFoldTraining = new Instances(instances);
            // if there is a remainder
            if ((numOfInstancesToSplit % (num_of_folds-fold)) != 0) {
                // add one more instance to the current fold
                currentFoldSize = predictedFoldSize + 1;
            } else {
                currentFoldSize = predictedFoldSize;
            }
            // the number of instances left to split
            numOfInstancesToSplit -= currentFoldSize;
            // creates the current fold
            currentFold = createFold(currentFoldTraining, foldLastIndex + 1 , currentFoldSize);
            // updating the index of the last instance that we added to the last fold
            foldLastIndex = foldLastIndex + currentFoldSize;
            buildClassifier(currentFoldTraining);
            // calculates the average error of the current fold
            sumOfErrors += calcAvgError(currentFold);
        }
        // return the average of all the errors
        return sumOfErrors/num_of_folds;
    }


    private Instances createFold (Instances instances, int firstIndex , int currentFoldSize){
        Instances fold = new Instances(instances, 0, 0);
        for (int i = 0; i < currentFoldSize ; i++) {
            fold.add(instances.remove(firstIndex));
        }

        //sets the training data => all instances - fold instances
        m_trainingInstances = instances;
        return fold;
    }
    /**
     * Finds the k nearest neighbors.
     * @param instance
     */
    public void findNearestNeighbors(Instance instance) {
        m_distances = new double[m_trainingInstances.numInstances()];
        m_kNearestNeighbors = new int[m_k];

        // sets the indexes to -1 (since we have a instance with index 0)
        for(int k = 0; k < m_kNearestNeighbors.length; k++)
        {
            m_kNearestNeighbors[k] = -1;
        }

        if(m_distanceCheck == DistanceCheck.Regular)
        {
            calcDistances(instance);
            // initialized with false
            boolean isUsed[] = new boolean[m_distances.length];
            // for k neighbors
            for (int neighborIndex = 0; neighborIndex < m_k; neighborIndex++ )
            {
                // for every instance in the training set
                for (int instanceDistance = 0; instanceDistance < m_distances.length; instanceDistance++)
                {
                    // if the instance was not taken as a neighbor yet
                    if (!isUsed[instanceDistance])
                    {
                        if (m_kNearestNeighbors[neighborIndex] != -1)
                        {
                            // if the current distance is smaller than the minimum distance found so far
                            if (m_distances[instanceDistance] < m_distances[m_kNearestNeighbors[neighborIndex]])
                            {
                                m_kNearestNeighbors[neighborIndex] = instanceDistance;
                            }
                        }
                        else
                        {
                            m_kNearestNeighbors[neighborIndex] = instanceDistance;
                        }
                    }
                }
                isUsed[m_kNearestNeighbors[neighborIndex]] = true;
            }
        }

        else
        {
            // fills the kNearestNeighbors array and sorts it by distance
            calcAndSortFirstKDistances(instance);
            double currentEffDistance;

            for (int i = m_k; i < m_trainingInstances.numInstances(); i++)
            {
                currentEffDistance = distanceCalculator.distance(instance, m_trainingInstances.instance(i), m_lpdistance, m_distances[m_kNearestNeighbors[m_k-1]]);
                // if the calculation wasn't stopped, we save the calculated distance
                // and insert the index of the instance to the neighbors array
                if(currentEffDistance != -1)
                {
                    m_distances[i] = currentEffDistance;
                    insertNewNeighbor(i);
                }
            }
        }
    }

    private void calcDistances(Instance instance){
        for (int i = 0; i < m_trainingInstances.numInstances(); i++){
           // lp distance is infinity
            m_distances[i] = distanceCalculator.distance(instance, m_trainingInstances.instance(i), m_lpdistance);
        }
    }

    private void calcAndSortFirstKDistances(Instance instance){
        boolean[] isUsed = new boolean[m_k];
        double minDistance;

        // inserts all the k first instances indexes to the neighbors array
        for (int i = 0; i < m_k; i++){
            m_distances[i] = distanceCalculator.distance(instance, m_trainingInstances.instance(i), m_lpdistance);
            m_kNearestNeighbors[i] = i;
        }

        // sorts the instances indexes by the distances from the tested instance
        for (int k = 0; k < m_k; k++)
        {
            minDistance = Double.MAX_VALUE;
            // finds the min value that wasn't already used
            for (int i = 0; i < m_kNearestNeighbors.length; i++)
            {
                if (!isUsed[i])
                {
                    if (m_distances[i] < minDistance)
                    {
                        m_kNearestNeighbors[k] = i;
                        minDistance = m_distances[i];
                    }
                }
            }
            isUsed[m_kNearestNeighbors[k]] = true;
        }
    }

    private void insertNewNeighbor(int index)
    {
        int tempIndex;
        // places the new neighbor's index in the k-1 index
        // (since its distance from the tested instance is smaller than current kth neighbor
        m_kNearestNeighbors[m_k - 1] = index;

        // inserts the new neighbor to its place in the sorted array
        // pushes the new neighbor from index K-1
        // no need to start from K-1 since the new neighbor is placed there already
        for (int i = m_k - 2 ; i >= 0; i--)
        {
            if (m_distances[index] < m_distances[m_kNearestNeighbors[i]])
            {
                tempIndex = m_kNearestNeighbors[i];
                m_kNearestNeighbors[i] = index;
                m_kNearestNeighbors[i+1] = tempIndex;
            }
            else
            {
                return;
            }
        }
    }

    /**
     * Calculates the average value of the given elements in the collection.
     * @param
     * @return
     */
    private double getAverageValue () {
       double sum = 0;
       int instanceIndex;
        // for every neighbor
       for(int i = 0; i < m_kNearestNeighbors.length; i++){
           instanceIndex = m_kNearestNeighbors[i];
           // currentWeight = 1/m_distances[instanceIndex];
           if (m_distances[instanceIndex] == 0)
           {
               return m_trainingInstances.instance(instanceIndex).classValue();
           }
           // get the class value of the neighbor
           sum += m_trainingInstances.instance(instanceIndex).classValue();
       }
        // return the average of the errors
       return sum/m_kNearestNeighbors.length;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     * @return
     */
    public double getWeightedAverageValue() {

        double sumDenominator = 0;
        double sumNuminator = 0;
        double currentWeight;
        double currentDistance;
        int instanceIndex;

        // for every neighbor
        for(int i = 0; i < m_kNearestNeighbors.length; i++){
            instanceIndex = m_kNearestNeighbors[i];
            // currentWeight = 1/m_distances[instanceIndex];
            if (m_distances[instanceIndex] == 0)
            {
                return m_trainingInstances.instance(instanceIndex).classValue();
            }

            // calculates current neighbor wi
            currentDistance = Math.pow(m_distances[instanceIndex],2);
            currentWeight = 1/currentDistance;
            // sums all wi
            sumDenominator += currentWeight;
            // sums all wi * f(xi)
            sumNuminator += (currentWeight * m_trainingInstances.instance(instanceIndex).classValue());
        }
        return sumNuminator/sumDenominator;
    }

    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}
