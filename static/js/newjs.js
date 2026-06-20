$(document).ready(function () {
    // Hide elements initially
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Function to preview the uploaded image
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr('src', e.target.result);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    // Show image preview on upload
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict button click event
    $('#btn-predict').click(function (event) {
        event.preventDefault(); // ✅ Prevents form submission

        var form_data = new FormData($('#upload-file')[0]);

        console.log("✅ Sending request to Flask...");

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Send AJAX request
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (data) {
                $('.loader').hide();
                $('#btn-predict').show();
                $('#result').fadeIn(600);

                // ✅ Extract result correctly
                if (data.result) {
                    $('#result').html('<strong>Result:</strong> ' + data.result);
                    console.log("✅ Prediction:", data.result);
                } else {
                    $('#result').html('<strong>Error:</strong> Invalid response');
                    console.log("❌ Error: Invalid response from server");
                }
            },
            error: function (error) {
                $('.loader').hide();
                $('#btn-predict').show();
                $('#result').fadeIn(600);
                $('#result').html('<strong>Error:</strong> ' + error.responseJSON.error);
                console.log("❌ Error:", error);
            }
        });
    });
});
