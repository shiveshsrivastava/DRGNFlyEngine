package com.example.drgnflyengine;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.WindowManager;

import com.example.drgnflyengine.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    private ExecutorService executorSingleThread = Executors.newSingleThreadExecutor();

    private static final int TARGET_WIDTH = 1280;

    private static final int TARGET_HEIGHT = 720;

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    Log.i("Permission: ", "Granted");
                    startCamera();
                } else {
                    Log.i("Permission: ", "Denied");
                }
    });

    // Used to load the 'drgnflyengine' library on application startup.
    static {
        System.loadLibrary("drgnflyengine");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }


    @Override
    protected void onDestroy() {
        executorSingleThread.shutdown();
        super.onDestroy();
    }

    private ImageAnalysis analyzeImages() {
        ResolutionSelector resolutionSelector =
                new ResolutionSelector.Builder().setResolutionStrategy(
                        new ResolutionStrategy(
                                new Size(TARGET_WIDTH, TARGET_HEIGHT), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER)).build();

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder().setResolutionSelector(resolutionSelector).setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build();

        //Passing image width and height dynamically, as it can change because of our FALLBACK_RULE_CLOSEST_HIGHER resolution strategy
        imageAnalysis.setAnalyzer(executorSingleThread, image -> {
            processFrames(image.getPlanes()[0].getBuffer(), image.getWidth(), image.getHeight());
            image.close();
        });

        return imageAnalysis;
    }

    private void startCamera() {

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                throw new RuntimeException(e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();

        CameraSelector cameraSelector =
                new CameraSelector.Builder().requireLensFacing(
                        CameraSelector.LENS_FACING_BACK).build();

        preview.setSurfaceProvider(binding.previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = analyzeImages();

        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageAnalysis);

        camera.getCameraControl().setLinearZoom(0.0f);

        binding.previewView.setImplementationMode(PreviewView.ImplementationMode.PERFORMANCE);
        binding.previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER);
    }

    /**
     * A native method that is implemented by the 'drgnflyengine' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    private native void processFrames(ByteBuffer pixelData, int width, int height);
}