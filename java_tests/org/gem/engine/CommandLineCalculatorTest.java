// package org.opensha.gem.GEM1.calc.gemCommandLineCalculator;
package org.gem.engine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.StringTokenizer;

import org.apache.commons.configuration.ConfigurationException;
import org.gem.engine.CalculatorConfigHelper.CalculationMode;
import org.gem.engine.CalculatorConfigHelper.ConfigItems;
import org.gem.engine.CalculatorConfigHelper.IntensityMeasure;
import org.gem.engine.hazard.memcached.BaseMemcachedTest;
import org.gem.engine.hazard.memcached.Cache;
import org.junit.Test;
import org.opensha.commons.data.Site;
import org.opensha.commons.geo.Location;

import com.google.gson.Gson;

public class CommandLineCalculatorTest extends BaseMemcachedTest {

    /**
     * The calculator does not yet run if it is give to calculate for an
     * intensity measure type "MMI". This tests veryfies that. It is expected to
     * fail.
     * 
     * @throws ConfigurationException
     */
    @Test(expected = IllegalArgumentException.class)
    public void testCalculatorConfig() throws ConfigurationException {

        /*
         * (state at 2010-10-07): This lets the test fail as expected
         * (2010-10-07)
         */
        // final String intensityMeasureTypeToTest = "MMI";
        final String intensityMeasureTypeToTest = IntensityMeasure.MMI.type();
        /*
         * 
         * (state at 2010-10-07): This would let the test end successfully.
         */
        // final String intensityMeasureTypeToTest =
        // IntensityMeasure.PGA.type();
        CommandLineCalculator clc =
                new CommandLineCalculator("CalculatorConfig.properties");
        clc.setConfigItem(ConfigItems.INTENSITY_MEASURE_TYPE.name(),
                intensityMeasureTypeToTest);
        clc.doCalculation();
    } // testCalculatorConfig()

    /**
     * Tests the probabilistic event based hazard calc through Monte Carlo logic
     * tree sampling.</br> If this test passes, i.e.</br> - all configurations
     * needed are given in the config file</br> - all needed objects of type
     * org.opensha.commons.param.Parameter are properly instantiated</br> - the
     * application workflow is not interrupted
     * 
     * @throws ConfigurationException
     */
    @Test
    public void testDoProbabilisticEventBasedCalcMonteCarlo()
            throws ConfigurationException {
        CommandLineCalculator clc =
                new CommandLineCalculator("CalculatorConfig.properties");
        String key = CalculatorConfigHelper.ConfigItems.CALCULATION_MODE.name();
        String mode = CalculationMode.MONTE_CARLO.value();
        // String calculationModeFull = CalculationMode.FULL.value();
        clc.setConfigItem(key, mode);
        testDoProbabilisticEventBasedCalc(clc);
    }

    /**
     * Tests the probabilistic event based hazard calc for the full logic tree
     * sampling.</br> If this test passes, i.e.</br> - all configurations needed
     * are given in the config file</br> - all needed objects of type
     * org.opensha.commons.param.Parameter are properly instantiated</br> - the
     * application workflow is not interrupted
     * 
     * @throws ConfigurationException
     */
    @Test
    public void testDoProbabilisticEventBasedCalcFull()
            throws ConfigurationException {
        CommandLineCalculator clc =
                new CommandLineCalculator("CalculatorConfig.properties");
        String key = CalculatorConfigHelper.ConfigItems.CALCULATION_MODE.name();
        String mode = CalculationMode.FULL.value();
        clc.setConfigItem(key, mode);
        testDoProbabilisticEventBasedCalc(clc);
    }

    /**
     * This method does the work for {@link
     * testDoProbabilisticEventBasedCalcMonteCarlo()} and {@link
     * testDoProbabilisticEventBasedCalcFull()}.
     * 
     * @param clc
     *            CommandLineCalculator object configured for either "full"
     *            event based hazard calculation or for the "Monte Carlo"
     *            approach.
     * @throws ConfigurationException
     */
    private void testDoProbabilisticEventBasedCalc(CommandLineCalculator clc)
            throws ConfigurationException {
        Map<Site, Double> result = clc.doCalculationProbabilisticEventBased();
        Object o = null;
        assertTrue(result != null);
        assertTrue(result instanceof Map);
        // assertTrue(result.size() > 0);
        assertTrue(result.size() > 0
                && (o = result.keySet().iterator().next()) instanceof Site);
        assertTrue(result.size() > 0 && result.get(o) instanceof Double);
    } // testDoProbabilisticEventBasedCalcThroughMonteCarloLogicTreeSampling()

    @Test
    // spike on the java.util.Properties object
            public
            void twoPropertiesAreEqualWithTheSameParameters() {
        Properties config1 = new Properties();
        config1.setProperty("KEY", "VALUE");

        Properties config2 = new Properties();
        config2.setProperty("KEY", "VALUE");

        Properties config3 = new Properties();
        config3.setProperty("ANOTHER_KEY", "ANOTHER_VALUE");

        Properties config4 = new Properties();
        config4.setProperty("KEY", "VALUE");
        config4.setProperty("ANOTHER_KEY", "ANOTHER_VALUE");

        assertTrue(config1.equals(config2));
        assertFalse(config1.equals(config3));
        assertFalse(config1.equals(config4));
        assertFalse(config3.equals(config4));
    }

    @Test
    public void twoCalculatorsAreEqualWithTheSameConfig() {
        Properties config1 = new Properties();
        config1.setProperty("KEY", "VALUE");

        Properties config2 = new Properties();
        config2.setProperty("ANOTHER_KEY", "ANOTHER_VALUE");

        CommandLineCalculator calc1 = new CommandLineCalculator(config1);
        CommandLineCalculator calc2 = new CommandLineCalculator(config1);
        CommandLineCalculator calc3 = new CommandLineCalculator(config2);

        assertTrue(calc1.equals(calc2));
        assertFalse(calc1.equals(calc3));
    }

    @Test
    public void supportsConfigurationReadingFromCache() {
        Properties config = new Properties();
        config.setProperty("KEY", "VALUE");
        config.setProperty("ANOTHER_KEY", "ANOTHER_VALUE");

        client.set("KEY", EXPIRE_TIME, new Gson().toJson(config));

        assertEquals(new CommandLineCalculator(config),
                new CommandLineCalculator(new Cache(LOCALHOST, PORT), "KEY"));
    }

    @Test
    public void peerSet1Case5() throws ConfigurationException {
        CommandLineCalculator clc =
                new CommandLineCalculator(
                        "tests/data/peerSet1Case5/CalculatorConfig.properties");
        clc.doCalculation();
        Map<Location, double[]> computedResults = readComputedResults();
        Map<Location, double[]> expectedResults =
                getHandSolutionsPeerTestSet1Case5();
        for (Location loc : expectedResults.keySet()) {
            for (int i = 0; i < expectedResults.get(loc).length; i++) {
                assertEquals(expectedResults.get(loc)[i],
                        computedResults.get(loc)[i], 1e-3);
            }
        }

    }

    private Map<Location, double[]> readComputedResults() {
        Map<Location, double[]> computedResults =
                new HashMap<Location, double[]>();
        File results = new File("build/individualHazardCurves.dat");
        FileReader fReader = null;
        try {
            fReader = new FileReader(results.getAbsolutePath());
            BufferedReader reader = new BufferedReader(fReader);
            String line = reader.readLine();
            StringTokenizer st = null;
            Location loc = null;
            double lon = Double.NaN;
            double lat = Double.NaN;
            double[] probEx = null;
            while ((line = reader.readLine()) != null) {
                st = new StringTokenizer(line);
                probEx = new double[st.countTokens() - 2];
                lon = Double.valueOf(st.nextToken());
                lat = Double.valueOf(st.nextToken());
                loc = new Location(lat, lon);
                int indexToken = 0;
                while (st.hasMoreTokens()) {
                    probEx[indexToken] = Double.valueOf(st.nextToken());
                    indexToken = indexToken + 1;
                }
                computedResults.put(loc, probEx);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return computedResults;
    }

    private Map<Location, double[]> getHandSolutionsPeerTestSet1Case5() {
        Map<Location, double[]> handSolutions =
                new HashMap<Location, double[]>();
        Location loc = new Location(38.000, -122.000);
        double[] probEx =
                new double[] { 3.99E-02, 3.99E-02, 3.98E-02, 2.99E-02,
                        2.00E-02, 1.30E-02, 8.58E-03, 5.72E-03, 3.88E-03,
                        2.69E-03, 1.91E-03, 1.37E-03, 9.74E-04, 6.75E-04,
                        2.52E-04, 0.00E+00 };
        handSolutions.put(loc, probEx);
        loc = new Location(37.910, -122.000);
        probEx =
                new double[] { 3.99E-02, 3.99E-02, 3.14E-02, 1.21E-02,
                        4.41E-03, 1.89E-03, 7.53E-04, 1.25E-04, 0.00E+00,
                        0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00,
                        0.00E+00, 0.00E+00 };
        handSolutions.put(loc, probEx);
        loc = new Location(38.225, -122.000);
        probEx =
                new double[] { 3.99E-02, 3.99E-02, 3.98E-02, 2.99E-02,
                        2.00E-02, 1.30E-02, 8.58E-03, 5.72E-03, 3.88E-03,
                        2.69E-03, 1.91E-03, 1.37E-03, 9.74E-04, 6.75E-04,
                        2.52E-04, 0.00E+00 };
        handSolutions.put(loc, probEx);
        return handSolutions;
    }
} // class CommandLineCalculatorTest
